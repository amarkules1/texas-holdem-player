"""
Flask Web Application backend for Texas Hold'em ML Advisor.
"""

import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.predictor import HoldemPredictor

app = Flask(__name__)
CORS(app)

# Initialize ML Predictor engine
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "holdem_policy_net.pt")
predictor = HoldemPredictor(model_path=MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON body"}), 400

        hole_cards = data.get("hole_cards", [])
        if len(hole_cards) != 2:
            return jsonify({"status": "error", "message": "Hole cards must contain exactly 2 cards"}), 400

        community_cards = data.get("community_cards", [])
        pot = float(data.get("pot", 100))
        all_day = float(data.get("all_day", 20))
        round_bet = float(data.get("round_bet", 0))
        stack = float(data.get("stack", 1000))
        position = int(data.get("position", 0))
        player_ct = int(data.get("player_ct", 6))
        big_blind = float(data.get("big_blind", 20))

        result = predictor.predict_scenario(
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot=pot,
            all_day=all_day,
            round_bet=round_bet,
            stack=stack,
            position=position,
            player_ct=player_ct,
            big_blind=big_blind
        )

        return jsonify({"status": "success", "data": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/presets", methods=["GET"])
def presets():
    preset_list = [
        {
            "id": "preflop_aks",
            "name": "Preflop Premium (A♠ K♠ in Cutoff)",
            "hole_cards": ["As", "Ks"],
            "community_cards": [],
            "pot": 50,
            "all_day": 20,
            "round_bet": 0,
            "stack": 1000,
            "position": 4,
            "player_ct": 6,
            "big_blind": 20
        },
        {
            "id": "flop_flush_draw",
            "name": "Flop Monster Draw (A♠ K♠ on Q♠ J♠ 2♥)",
            "hole_cards": ["As", "Ks"],
            "community_cards": ["Qs", "Js", "2h"],
            "pot": 150,
            "all_day": 40,
            "round_bet": 0,
            "stack": 980,
            "position": 2,
            "player_ct": 4,
            "big_blind": 20
        },
        {
            "id": "turn_full_house",
            "name": "Turn Full House (Q♠ Q♦ on Q♥ J♠ J♣ 5♦)",
            "hole_cards": ["Qs", "Qd"],
            "community_cards": ["Qh", "Js", "Jc", "5d"],
            "pot": 350,
            "all_day": 80,
            "round_bet": 0,
            "stack": 900,
            "position": 1,
            "player_ct": 3,
            "big_blind": 20
        },
        {
            "id": "short_stack_tens",
            "name": "Short Stack 10♣ 10♦ Facing Raise",
            "hole_cards": ["10c", "10d"],
            "community_cards": [],
            "pot": 120,
            "all_day": 80,
            "round_bet": 20,
            "stack": 250,
            "position": 1,
            "player_ct": 6,
            "big_blind": 20
        },
        {
            "id": "river_weak_ace",
            "name": "River Weak Ace Facing Large Bet",
            "hole_cards": ["Ah", "5h"],
            "community_cards": ["Kd", "10s", "8c", "2h", "Jd"],
            "pot": 400,
            "all_day": 200,
            "round_bet": 0,
            "stack": 800,
            "position": 0,
            "player_ct": 2,
            "big_blind": 20
        }
    ]
    return jsonify({"status": "success", "presets": preset_list})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
