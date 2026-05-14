from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field


MOVES = ("tas", "kagit", "makas")
DISPLAY = {
    "tas": "Tas",
    "kagit": "Kagit",
    "makas": "Makas",
}

# Her hamlenin yendigi hamle.
BEATS = {
    "tas": "makas",
    "kagit": "tas",
    "makas": "kagit",
}

# Kullanicinin tahmin edilen hamlesini yenen bot hamlesi.
COUNTER_TO = {target: move for move, target in BEATS.items()}

ALIASES = {
    "t": "tas",
    "tas": "tas",
    "taş": "tas",
    "k": "kagit",
    "ka": "kagit",
    "kagit": "kagit",
    "kağıt": "kagit",
    "m": "makas",
    "makas": "makas",
}


@dataclass(frozen=True)
class Prediction:
    method: str
    move: str
    confidence: float


@dataclass
class AdaptiveRPSBot:
    user_history: list[str] = field(default_factory=list)
    bot_history: list[str] = field(default_factory=list)
    method_scores: dict[str, float] = field(
        default_factory=lambda: {
            "repeat_last": 1.0,
            "global_frequency": 1.0,
            "recent_frequency": 1.0,
            "markov_1": 1.0,
            "markov_2": 1.0,
            "markov_3": 1.0,
            "response_to_bot": 1.0,
            "cycle_forward": 0.6,
            "cycle_backward": 0.6,
        }
    )
    pending_predictions: dict[str, str] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)

    def choose_bot_move(self) -> tuple[str, str, str, float]:
        """Tahmin edilen kullanici hamlesini ve onu yenen bot hamlesini dondurur."""
        predictions = self._collect_predictions()
        self.pending_predictions = {
            prediction.method: prediction.move for prediction in predictions
        }

        if not predictions:
            predicted_user_move = self.rng.choice(MOVES)
            bot_move = COUNTER_TO[predicted_user_move]
            return bot_move, predicted_user_move, "random", 0.0

        best = max(
            predictions,
            key=lambda prediction: (
                self.method_scores.get(prediction.method, 1.0) * prediction.confidence,
                prediction.confidence,
            ),
        )
        bot_move = COUNTER_TO[best.move]
        return bot_move, best.move, best.method, best.confidence

    def observe_round(self, user_move: str, bot_move: str) -> None:
        """Son turdan sonra tahmin yontemlerinin agirligini gunceller."""
        for method, predicted_move in self.pending_predictions.items():
            old_score = self.method_scores.get(method, 1.0)
            if predicted_move == user_move:
                self.method_scores[method] = old_score + 1.0
            else:
                self.method_scores[method] = max(0.2, old_score - 0.15)

        self.user_history.append(user_move)
        self.bot_history.append(bot_move)
        self.pending_predictions.clear()

    def _collect_predictions(self) -> list[Prediction]:
        predictions: list[Prediction] = []

        if not self.user_history:
            return predictions

        predictions.append(Prediction("repeat_last", self.user_history[-1], 0.55))

        global_frequency = self._frequency_prediction(
            self.user_history, "global_frequency"
        )
        if global_frequency:
            predictions.append(global_frequency)

        recent_frequency = self._frequency_prediction(
            self.user_history[-6:], "recent_frequency"
        )
        if recent_frequency:
            predictions.append(recent_frequency)

        for order in (1, 2, 3):
            markov_prediction = self._markov_prediction(order)
            if markov_prediction:
                predictions.append(markov_prediction)

        response_prediction = self._response_to_bot_prediction()
        if response_prediction:
            predictions.append(response_prediction)

        predictions.extend(self._cycle_predictions())
        return predictions

    def _frequency_prediction(
        self, moves: list[str], method: str
    ) -> Prediction | None:
        if not moves:
            return None

        counts = Counter(moves)
        move, count = self._random_top_count(counts)
        confidence = count / len(moves)
        return Prediction(method, move, confidence)

    def _markov_prediction(self, order: int) -> Prediction | None:
        if len(self.user_history) <= order:
            return None

        suffix = tuple(self.user_history[-order:])
        counts: Counter[str] = Counter()

        for index in range(len(self.user_history) - order):
            state = tuple(self.user_history[index : index + order])
            next_move = self.user_history[index + order]
            if state == suffix:
                counts[next_move] += 1

        if not counts:
            return None

        move, count = self._random_top_count(counts)
        samples = sum(counts.values())
        confidence = (count / samples) * min(1.0, samples / 3)
        return Prediction(f"markov_{order}", move, confidence)

    def _response_to_bot_prediction(self) -> Prediction | None:
        if not self.bot_history:
            return None

        last_bot_move = self.bot_history[-1]
        counts: Counter[str] = Counter()

        for index, bot_move in enumerate(self.bot_history[:-1]):
            if bot_move == last_bot_move:
                counts[self.user_history[index + 1]] += 1

        if not counts:
            return None

        move, count = self._random_top_count(counts)
        samples = sum(counts.values())
        confidence = (count / samples) * min(1.0, samples / 3)
        return Prediction("response_to_bot", move, confidence)

    def _cycle_predictions(self) -> list[Prediction]:
        last_move = self.user_history[-1]
        index = MOVES.index(last_move)
        return [
            Prediction("cycle_forward", MOVES[(index + 1) % len(MOVES)], 0.35),
            Prediction("cycle_backward", MOVES[(index - 1) % len(MOVES)], 0.35),
        ]

    def _random_top_count(self, counts: Counter[str]) -> tuple[str, int]:
        highest = max(counts.values())
        tied_moves = [move for move, count in counts.items() if count == highest]
        move = self.rng.choice(tied_moves)
        return move, highest


def normalize_move(raw_move: str) -> str | None:
    text = raw_move.strip().lower()
    return ALIASES.get(text)


def round_result(user_move: str, bot_move: str) -> str:
    if user_move == bot_move:
        return "draw"
    if BEATS[user_move] == bot_move:
        return "user"
    return "bot"


def print_round_summary(
    user_move: str,
    bot_move: str,
    predicted_move: str,
    method: str,
    confidence: float,
    result: str,
) -> None:
    print(f"Sen: {DISPLAY[user_move]} | Bot: {DISPLAY[bot_move]}")
    print(
        "Bot tahmini: "
        f"{DISPLAY[predicted_move]} ({method}, guven: %{confidence * 100:.0f})"
    )

    if result == "draw":
        print("Sonuc: Berabere.")
    elif result == "user":
        print("Sonuc: Sen kazandin.")
    else:
        print("Sonuc: Bot kazandi.")


def main() -> None:
    bot = AdaptiveRPSBot()
    score = Counter()

    print("Tahminli Tas-Kagit-Makas Botu")
    print("Hamleler: tas/t, kagit/k, makas/m. Cikmak icin q yaz.")

    round_number = 1
    while True:
        bot_move, predicted_move, method, confidence = bot.choose_bot_move()
        raw_move = input(f"\n{round_number}. tur - hamlen: ")

        if raw_move.strip().lower() in {"q", "quit", "exit", "cikis", "çıkış"}:
            break

        user_move = normalize_move(raw_move)
        if user_move is None:
            print("Gecersiz hamle. Ornek: tas, kagit, makas")
            continue

        result = round_result(user_move, bot_move)
        score[result] += 1
        bot.observe_round(user_move, bot_move)

        print_round_summary(
            user_move=user_move,
            bot_move=bot_move,
            predicted_move=predicted_move,
            method=method,
            confidence=confidence,
            result=result,
        )
        print(
            "Skor -> "
            f"Sen: {score['user']} | Bot: {score['bot']} | Berabere: {score['draw']}"
        )
        round_number += 1

    print(
        "\nFinal skor -> "
        f"Sen: {score['user']} | Bot: {score['bot']} | Berabere: {score['draw']}"
    )


if __name__ == "__main__":
    main()
