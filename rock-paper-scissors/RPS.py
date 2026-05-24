import random

def player(prev_play, opponent_history=[]):

    opponent_history.append(prev_play)

    if prev_play == "":
        opponent_history.clear()
        return random.choice(["R", "P", "S"])

    last_ten = opponent_history[-10:]

    most_common = max(set(last_ten), key=last_ten.count)

    counter = {
        "R": "P",
        "P": "S",
        "S": "R"
    }

    return counter[most_common]