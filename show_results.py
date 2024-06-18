import pandas as pd
import numpy as np
import plotly.express as px
import datasets
import tiktoken
import datetime
import argparse
import os
import math
import prettytable as pt

from glob import glob
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from collections import defaultdict, Counter
from bench_utils import load_model_answers, model_name_to_id, load_image_categoeis, load_question_categoeis, load_model_judgements
from functools import partial

def compute_mle_elo(df, baseline, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X,Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as gpt-4-0314 = 1000
    if baseline in models.index:
        elo_scores += 1000 - elo_scores[models[baseline]]
    return pd.Series(elo_scores, index = models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    rows.append(func_compute_elo(battles)) # add the original elo as the first row
    for i in tqdm(range(num_round), desc="bootstrap"):
        try:
            rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
        except Exception as e:
            print("two few battles, error in bootstrap, break")
            break
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def preety_print_two_ratings(ratings_1, ratings_2, column_names):
    df = pd.DataFrame([
        [n, ratings_1[n], ratings_2[n]] for n in ratings_1.keys()
    ], columns=["Model", column_names[0], column_names[1]]).sort_values(column_names[0], ascending=False).reset_index(drop=True)
    df[column_names[0]] = (df[column_names[0]] + 0.5).astype(int)
    df[column_names[1]] = (df[column_names[1]] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def visualize_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        rating = df.quantile(.5),
        upper = df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y",
                     error_y_minus="error_y_minus", text="rating_rounded",
                     title=title)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating",
                      height=600)
    return fig


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.nan for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def get_win_rate_column(df, column, baseline):
    to_dict = df[["model", column]].set_index("model").to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    print(win_rate_table)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x * 100, 2))


def get_battles_from_judgement(judge_name, baseline, model_judgements, first_game_only=False, WEIGHT=3, bench_name="vision_bench", ):
    arena_hard_battles = pd.DataFrame()
    
    print("Turning judgement results into battles...")

    # directory = f"data/{bench_name}/model_judgement/{judge_name}"
    # assert os.path.exists(directory)
    # for file in tqdm(glob(f"{directory}/*jsonl")):
    #     df = pd.read_json(file, lines=True)
    for model in model_judgements:
        df = pd.DataFrame.from_dict(model_judgements[model], orient="index")

        for _, row in df.iterrows():
            # game 1
            output = {"question_id": row["question_id"],
                    "model_a": baseline,
                    "model_b": row["model"]}

            game = row["games"][0]

            weight = 1
            if game["score"] == "A=B":
                output["winner"] = "tie"
            elif game["score"] == "A>B":
                output["winner"] = "model_a"
            elif game["score"] == "A>>B":
                output["winner"] = "model_a"
                weight = WEIGHT
            elif game["score"] == "B>A":
                output["winner"] = "model_b"
            elif game["score"] == "B>>A":
                output["winner"] = "model_b"
                weight = WEIGHT
            else:
                weight = 0

            if weight:
                arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])

            if not first_game_only and len(row["games"]) > 1:
                # game 2
                output = {"question_id": row["question_id"],
                        "model_a": baseline,
                        "model_b": row["model"]}

                game = row["games"][1]

                weight = 1
                if game["score"] == "A=B":
                    output["winner"] = "tie"
                elif game["score"] == "A>B":
                    output["winner"] = "model_b"
                elif game["score"] == "A>>B":
                    output["winner"] = "model_b"
                    weight = WEIGHT
                elif game["score"] == "B>A":
                    output["winner"] = "model_a"
                elif game["score"] == "B>>A":
                    output["winner"] = "model_a"
                    weight = WEIGHT
                else:
                    weight = 0

                if weight:
                    arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])
    arena_hard_battles.to_json(f"data/{bench_name}_battles.jsonl", lines=True, orient="records")
    return arena_hard_battles

def get_reward_from_judgement(model_judgements, first_game_only=False):
    print("Turning judgement results into battles...")
    model_reward_info = {}
    for model in model_judgements:
        df = pd.DataFrame.from_dict(model_judgements[model], orient="index")

        total_rewards = []
        total_is_win = []
        total_is_win_or_tie = []
        counters = {
            "tie": 0,
            "better": 0,
            "much better": 0,
            "worse": 0,
            "much worse": 0
        }
        for _, row in df.iterrows():
            # game 1
            game = row["games"][0]

            reward = 0
            if game["score"] == "A=B":
                counters["tie"] += 1
                reward -= 0
            elif game["score"] == "A>B":
                counters["worse"] += 1
                reward -= 50
            elif game["score"] == "A>>B":
                counters["much worse"] += 1
                reward -= 100
            elif game["score"] == "B>A":
                counters["better"] += 1
                reward -= -50
            elif game["score"] == "B>>A":
                counters["much better"] += 1
                reward -= -100
            else:
                counters["tie"] += 1
                reward -= 0

            if not first_game_only and len(row["games"]) > 1:
                game = row["games"][1]

                if game["score"] == "A=B":
                    counters["tie"] += 1
                    reward += 0
                elif game["score"] == "A>B":
                    counters["better"] += 1
                    reward += 50
                elif game["score"] == "A>>B":
                    counters["much better"] += 1
                    reward += 100
                elif game["score"] == "B>A":
                    counters["worse"] += 1
                    reward += -50
                elif game["score"] == "B>>A":
                    counters["much worse"] += 1
                    reward += -100
                else:
                    counters["tie"] += 1
                    reward += 0
            total_is_win.append(reward > 0)
            total_is_win_or_tie.append(reward >= 0)
            total_rewards.append(reward)
        model_reward_info[model] = {
            "reward": np.mean(total_rewards),
            "win_rate": np.mean(total_is_win),
            "win_or_tie_rate": np.mean(total_is_win_or_tie),
            "vote_type_counts": counters
        }
    
    return model_reward_info

def run_elo_simulation(model_answers, model_judgements, args):
    question_len = len(list(model_answers.values())[0])
    if question_len < 10:
        print(f"Too few questions ({question_len} < 10), skip")
        return 
    if args.load_battles:
        assert os.path.exists(f"data/{args.bench_name}_battles.jsonl")
        battles = pd.read_json(f"data/{args.bench_name}_battles.jsonl", lines=True)
    else:
        battles = get_battles_from_judgement(args.judge_name, args.baseline, model_judgements, args.first_game_only, args.weight)
    
    model_reward_info = get_reward_from_judgement(model_judgements, args.first_game_only)
        
    bootstrap_online_elo = compute_mle_elo(battles, args.baseline)
    _compute_mle_elo = partial(compute_mle_elo, baseline=args.baseline)


    if args.load_bootstrap:
        bootstrap_elo_lu = pd.read_json("data/bootstrapping_results.jsonl", lines=True)
    else:
        np.random.seed(42)
        bootstrap_elo_lu = get_bootstrap_result(battles, _compute_mle_elo, args.num_rounds)
        bootstrap_elo_lu.to_json("data/bootstrapping_results.jsonl", lines=True, orient="records")

    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats['results'].astype('object')

    for i, model in enumerate(bootstrap_online_elo.index):
        assert model in bootstrap_elo_lu.columns
        model_id = model_name_to_id(model)
        stats.at[i, "model"] = model
        stats.at[i, "score"] = bootstrap_online_elo[model]
        stats.at[i, "lower"] = np.percentile(bootstrap_elo_lu[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_elo_lu[model], 97.5)
        stats.at[i, "reward"] = model_reward_info[model_id]["reward"]
        stats.at[i, "win_rate"] = model_reward_info[model_id]["win_rate"]
        stats.at[i, "win_or_tie_rate"] = model_reward_info[model_id]["win_or_tie_rate"]
        stats.at[i, "much_better"] = model_reward_info[model_id]["vote_type_counts"]["much better"]
        stats.at[i, "better"] = model_reward_info[model_id]["vote_type_counts"]["better"]
        stats.at[i, "tie"] = model_reward_info[model_id]["vote_type_counts"]["tie"]
        stats.at[i, "worse"] = model_reward_info[model_id]["vote_type_counts"]["worse"]
        stats.at[i, "much_worse"] = model_reward_info[model_id]["vote_type_counts"]["much worse"]

        length = 0
        model_id = model_name_to_id(model)
        if model_id in model_answers:
            for _, row in model_answers[model_id].items():
                # turn = row["choices"][0]["turns"][0]
                length += row["token_len"]
            length /= len(model_answers[model_id])
            
        stats.at[i, "avg_tokens"] = int(length)
        stats.at[i, "results"] = bootstrap_elo_lu[model].tolist()
    
    if not args.show_elo:
        stats.sort_values(by="model", inplace=True)
        print("Original ELO Winning Table: ")
        stats["score"] = get_win_rate_column(stats, "score", args.baseline).tolist()
        print("(95% CI) Upper Bound ELO Winning Table: ")
        stats["lower"] = get_win_rate_column(stats, "lower", args.baseline).tolist()
        print("(95% CI) Lower Bound ELO Winning Table: ")
        stats["upper"] = get_win_rate_column(stats, "upper", args.baseline).tolist()
        decimal = 1
    else:
        decimal = 0
        stats = stats.astype({"score" : int, "lower" : int, "upper" : int})
    
    print("Simulated Elo Leaderboard: ")
    stats.sort_values(by="score", ascending=False, inplace=True)
    
    pt_table = pt.PrettyTable()
    pt_table.field_names = ["Model", "Score", "95% CI", "Win Rate", "Reward", "Much Better", "Better", "Tie", "Worse", "Much Worse", "Avg Tokens"]
    for _, row in stats.iterrows():
        interval = str((round(row['lower'] - row['score'], decimal), round(row['upper'] - row['score'], decimal)))
        # pt_table.add_row([row['model'], row['score'], interval, round(row['win_rate'] * 100, 2), round(row['win_or_tie_rate'] * 100, 2), row['reward'], int(row['avg_tokens'])])
        pt_table.add_row([row['model'], row['score'], interval, f"{round(row['win_rate'] * 100, 2)}%", row['reward'], row['much_better'], row['better'], row['tie'], row['worse'], row['much_worse'], int(row['avg_tokens'])])
    print(pt_table)
    
    # for _, row in stats.iterrows():
    #     interval = str((round(row['lower'] - row['score'], decimal), round(row['upper'] - row['score'], decimal)))
    #     to_print_line = f"{row['model'] : <30} | score: {round(row['score'], decimal) : ^5} | 95% CI: {interval : ^12}"
    #     to_print_line += f" | win_rate: {round(row['win_rate'] * 100, 2) : ^5}% | win_or_tie_rate: {round(row['win_or_tie_rate'] * 100, 2) : ^5}% | reward: {round(row['reward'], decimal) : ^5}"
    #     to_print_line += f" | average #tokens: {int(row['avg_tokens'])}"
    #     print(to_print_line)

    # save dict
    model_score_dict = {}
    for _, row in stats.iterrows():
        model_score_dict[row["model"]] = row["score"]
    print(model_score_dict)
    if args.output:
        cur_date = datetime.datetime.now()
        date_str = cur_date.strftime("%Y%m%d")
        stats.to_json(f"{args.bench_name}_leaderboard_{date_str}.json", orient="records", indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="vision_bench")
    parser.add_argument("--judge-name", type=str, default="gpt-4o")
    parser.add_argument("--baseline", type=str, default="claude-3-sonnet-20240229")
    parser.add_argument("--load-battles", action="store_true")
    parser.add_argument("--load-bootstrap", action="store_true")
    parser.add_argument("--show-elo", action="store_true")
    parser.add_argument("--weight", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--output", action="store_true")
    parser.add_argument("--first-game-only", action="store_true")
    parser.add_argument("--divide-by-category", action="store_true")
    args = parser.parse_args()
    print(args)
    assert not args.load_bootstrap or (args.load_battles and args.load_bootstrap), "If loading prexisting bootstrapping data, you must also load preexisting battles."

    answer_dir = os.path.join("data", args.bench_name, "model_answers")
    all_model_answers = load_model_answers(answer_dir)
    
    judgements_dir = os.path.join("data", args.bench_name, "model_judgements", f"judge_{args.judge_name}_reference_{args.baseline}")
    all_model_judgements = load_model_judgements(judgements_dir, SAMPLE_START=0, MAX_SAMPLE_BENCH_SIZE=500)

    if not args.divide_by_category:
        run_elo_simulation(all_model_answers, all_model_judgements, args)
    else:
        # load question categories and image categories
        question_categories = load_question_categoeis(args.bench_name)
        image_categories = load_image_categoeis(args.bench_name)
        question_categories_counter = Counter(question_categories.values())
        image_categories_counter = Counter(image_categories.values())
        # sort
        question_categories_counter = dict(sorted(question_categories_counter.items(), key=lambda item: item[0]))
        image_categories_counter = dict(sorted(image_categories_counter.items(), key=lambda item: item[0]))
        question_category_set = list(question_categories_counter.keys())
        image_category_set = list(image_categories_counter.keys())
        
        print("All question categories: ")
        for x, y in question_categories_counter.items():
            print("-", x, ":", y)
        print("All image categories: ")
        for x, y in image_categories_counter.items():
            print("-", x, ":", y)
            
        for question_category in question_category_set:
            print("\n\n=====================================")
            print(f"Question Category: {question_category}")
            filtered_answers = defaultdict(dict)
            for model_id, model_answers in all_model_answers.items():
                for question_id, answer in model_answers.items():
                    if question_categories[question_id] == question_category:
                        filtered_answers[model_id][question_id] = answer
            filtered_judgements = defaultdict(dict)
            for model_id, model_judgements in all_model_judgements.items():
                for question_id, judgement in model_judgements.items():
                    if question_categories[question_id] == question_category:
                        filtered_judgements[model_id][question_id] = judgement
            run_elo_simulation(filtered_answers, filtered_judgements, args)
        for image_category in image_category_set:
            print("\n\n=====================================")
            print(f"Image Category: {image_category}")
            filtered_answers = defaultdict(dict)
            for model_id, model_answers in all_model_answers.items():
                for question_id, answer in model_answers.items():
                    if image_categories[question_id] == image_category:
                        filtered_answers[model_id][question_id] = answer
            filtered_judgements = defaultdict(dict)
            for model_id, model_judgements in all_model_judgements.items():
                for question_id, judgement in model_judgements.items():
                    if image_categories[question_id] == image_category:
                        filtered_judgements[model_id][question_id] = judgement
            run_elo_simulation(filtered_answers, filtered_judgements, args)