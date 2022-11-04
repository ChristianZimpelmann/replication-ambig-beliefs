import pandas as pd


def calculate_matching_probabilites(choices):
    def get_matching_prob(choice_and_prob):
        """
        Calculates the baseline matching probability.
        args:
            choice_and_prob: tuple, containing string and integer
        returns:
            pandas interval
        """
        # print(choice_and_prob[1], choice_and_prob[0])
        dic = {
            1: {
                True: pd.Interval(left=1, right=5, closed="both"),
                False: pd.Interval(left=0, right=1, closed="both"),
            },
            5: {True: pd.Interval(left=5, right=10, closed="both")},
            20: {
                True: pd.Interval(left=20, right=30, closed="both"),
                False: pd.Interval(left=10, right=20, closed="both"),
            },
            40: {
                True: pd.Interval(left=40, right=50, closed="both"),
                False: pd.Interval(left=30, right=40, closed="both"),
            },
            60: {
                True: pd.Interval(left=60, right=70, closed="both"),
                False: pd.Interval(left=50, right=60, closed="both"),
            },
            80: {
                True: pd.Interval(left=80, right=90, closed="both"),
                False: pd.Interval(left=70, right=80, closed="both"),
            },
            95: {False: pd.Interval(left=90, right=95, closed="both")},
            99: {
                True: pd.Interval(left=99, right=100, closed="both"),
                False: pd.Interval(left=95, right=99, closed="both"),
            },
        }
        return dic[choice_and_prob["lottery_p_win"].values[0]][
            choice_and_prob["choice"].values[0]
        ]

    def select_final_nodes(df):
        """
        Extracts the final choice and node from the choices dataframe
        args:
            df: pandas dataframe
        returns:
            tuple, containing string and integer
        """
        end_nodes = [1, 5, 20, 40, 60, 80, 95, 99]

        # Ignore answers to extra questions
        df = df.copy().query("choice_final == False")

        # Select only rows at end nodes
        df = df[df[["lottery_p_win"]].isin(end_nodes).values]
        df = df.query("~(choice==False & lottery_p_win==5)")
        df = df.query("~(choice==True & lottery_p_win==95)")

        return df

    final_nodes = select_final_nodes(choices)
    # Calculate baseline matching probability
    matching_prob_interval = final_nodes.groupby(["personal_id", "wave", "aex_event"])[
        ["choice", "lottery_p_win"]
    ].apply(get_matching_prob)
    # Put interesting stuff in a dataframe
    df = pd.DataFrame(
        matching_prob_interval, columns=["baseline_matching_prob_interval"]
    )
    df["baseline_matching_prob_leftb"] = matching_prob_interval.apply(lambda x: x.left)
    df["baseline_matching_prob_midp"] = matching_prob_interval.apply(lambda x: x.mid)
    df["baseline_matching_prob_rightb"] = matching_prob_interval.apply(
        lambda x: x.right
    )

    return df
