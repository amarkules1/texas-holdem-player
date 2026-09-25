/**
 * Texas Hold'em ML Advisor - Vanilla JS Application Logic
 */

// Application State
const state = {
    holeCards: [null, null],
    communityCards: [null, null, null, null, null],
    activePicker: null, // { type: 'hole'|'comm', index: number }
    presets: []
};

// Card definitions
const RANKS = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"];
const SUITS = [
    { code: "s", name: "Spades", symbol: "♠", class: "suit-spades" },
    { code: "h", name: "Hearts", symbol: "♥", class: "suit-hearts" },
    { code: "d", name: "Diamonds", symbol: "♦", class: "suit-diamonds" },
    { code: "c", name: "Clubs", symbol: "♣", class: "suit-clubs" }
];

document.addEventListener("DOMContentLoaded", () => {
    fetchPresets();
    buildCardModalGrid();
    
    // Default initial cards (A♠ K♠)
    setCard("hole", 0, "As");
    setCard("hole", 1, "Ks");
    setCard("comm", 0, "Qs");
    setCard("comm", 1, "Js");
    setCard("comm", 2, "2h");

    // Automatically analyze initial scenario
    analyzeScenario();
});

// Fetch presets from API
async function fetchPresets() {
    try {
        const response = await fetch("/api/presets");
        const json = await response.json();
        if (json.status === "success") {
            state.presets = json.presets;
            renderPresets();
        }
    } catch (err) {
        console.error("Failed to load presets:", err);
    }
}

// Render Preset Chips
function renderPresets() {
    const container = document.getElementById("presetContainer");
    container.innerHTML = "";
    state.presets.forEach(p => {
        const chip = document.createElement("button");
        chip.className = "preset-chip";
        chip.innerText = p.name;
        chip.onclick = () => loadPreset(p);
        container.appendChild(chip);
    });
}

// Load Preset Scenario
function loadPreset(preset) {
    state.holeCards = [null, null];
    state.communityCards = [null, null, null, null, null];

    preset.hole_cards.forEach((c, idx) => setCard("hole", idx, c));
    preset.community_cards.forEach((c, idx) => setCard("comm", idx, c));

    // Clear remaining slots
    for (let i = preset.community_cards.length; i < 5; i++) {
        clearCard("comm", i);
    }

    document.getElementById("potInput").value = preset.pot;
    document.getElementById("allDayInput").value = preset.all_day;
    document.getElementById("stackInput").value = preset.stack;
    document.getElementById("positionInput").value = preset.position;
    document.getElementById("playerCtInput").value = preset.player_ct;
    document.getElementById("bigBlindInput").value = preset.big_blind;

    analyzeScenario();
}

// Open Card Picker Modal
function openCardPicker(type, index) {
    state.activePicker = { type, index };
    const titleText = type === "hole" ? `Select Hole Card ${index + 1}` : `Select Community Card ${index + 1}`;
    document.getElementById("modalTitle").innerText = titleText;
    
    updateModalGridDisabledState();
    document.getElementById("cardModal").classList.add("active");
}

function closeCardModal() {
    document.getElementById("cardModal").classList.remove("active");
    state.activePicker = null;
}

function closeModalOnOverlay(e) {
    if (e.target.id === "cardModal") {
        closeCardModal();
    }
}

// Build 52-Card Modal Grid
function buildCardModalGrid() {
    const grid = document.getElementById("cardGrid");
    grid.innerHTML = "";

    SUITS.forEach(suit => {
        RANKS.forEach(rank => {
            const cardCode = `${rank}${suit.code}`;
            const cardEl = document.createElement("div");
            cardEl.className = "picker-card";
            cardEl.dataset.code = cardCode;
            cardEl.onclick = () => onCardSelected(cardCode);

            cardEl.innerHTML = `
                <span class="${suit.class}">${rank}</span>
                <span class="${suit.class}" style="align-self:center;font-size:16px;">${suit.symbol}</span>
            `;
            grid.appendChild(cardEl);
        });
    });
}

// Update Disabled State in Modal Grid
function updateModalGridDisabledState() {
    const selected = new Set([
        ...state.holeCards.filter(Boolean),
        ...state.communityCards.filter(Boolean)
    ]);

    // Exclude currently active slot card
    if (state.activePicker) {
        const currentVal = state.activePicker.type === "hole" 
            ? state.holeCards[state.activePicker.index] 
            : state.communityCards[state.activePicker.index];
        selected.delete(currentVal);
    }

    const pickerCards = document.querySelectorAll(".picker-card");
    pickerCards.forEach(el => {
        const code = el.dataset.code;
        if (selected.has(code)) {
            el.classList.add("disabled");
        } else {
            el.classList.remove("disabled");
        }
    });
}

// On Card Selected from Modal
function onCardSelected(cardCode) {
    if (!state.activePicker) return;
    setCard(state.activePicker.type, state.activePicker.index, cardCode);
    closeCardModal();
}

// Set Card Helper
function setCard(type, index, cardCode) {
    if (type === "hole") {
        state.holeCards[index] = cardCode;
        renderCardSlot(`hole${index}`, cardCode);
    } else {
        state.communityCards[index] = cardCode;
        renderCardSlot(`comm${index}`, cardCode);
    }
}

// Clear Card Helper
function clearCard(type, index) {
    if (type === "hole") {
        state.holeCards[index] = null;
        renderEmptySlot(`hole${index}`, `Card ${index + 1}`);
    } else {
        state.communityCards[index] = null;
        const stages = ["Flop 1", "Flop 2", "Flop 3", "Turn", "River"];
        renderEmptySlot(`comm${index}`, stages[index]);
    }
}

function clearCommunityCards() {
    for (let i = 0; i < 5; i++) {
        clearCard("comm", i);
    }
}

// Render Card Slot UI
function renderCardSlot(elementId, cardCode) {
    const slot = document.getElementById(elementId);
    if (!slot) return;

    if (!cardCode) {
        slot.className = "card-slot empty";
        return;
    }

    const rankStr = cardCode.slice(0, -1);
    const suitCode = cardCode.slice(-1).toLowerCase();
    const suitObj = SUITS.find(s => s.code === suitCode) || SUITS[0];

    slot.className = "card-slot filled";
    slot.innerHTML = `
        <div class="poker-card">
            <span class="card-rank ${suitObj.class}">${rankStr}</span>
            <span class="card-suit-large ${suitObj.class}">${suitObj.symbol}</span>
        </div>
    `;
}

function renderEmptySlot(elementId, desc) {
    const slot = document.getElementById(elementId);
    if (!slot) return;
    slot.className = "card-slot empty";
    slot.innerHTML = `
        <span class="plus-icon">+</span>
        <span class="slot-desc">${desc}</span>
    `;
}

// Reset All Form Fields
document.getElementById("resetBtn").addEventListener("click", () => {
    state.holeCards = [null, null];
    state.communityCards = [null, null, null, null, null];
    for (let i = 0; i < 2; i++) clearCard("hole", i);
    for (let i = 0; i < 5; i++) clearCard("comm", i);
});

// Analyze Scenario via API
async function analyzeScenario() {
    const holeValid = state.holeCards.filter(Boolean);
    if (holeValid.length !== 2) {
        alert("Please select exactly 2 Hole Cards.");
        return;
    }

    const commValid = state.communityCards.filter(Boolean);

    const payload = {
        hole_cards: holeValid,
        community_cards: commValid,
        pot: parseFloat(document.getElementById("potInput").value) || 100,
        all_day: parseFloat(document.getElementById("allDayInput").value) || 20,
        round_bet: 0,
        stack: parseFloat(document.getElementById("stackInput").value) || 1000,
        position: parseInt(document.getElementById("positionInput").value) || 0,
        player_ct: parseInt(document.getElementById("playerCtInput").value) || 6,
        big_blind: parseFloat(document.getElementById("bigBlindInput").value) || 20
    };

    const btn = document.getElementById("analyzeBtn");
    btn.disabled = true;
    btn.innerHTML = `<span class="btn-sparkle">⏳</span> ANALYZING SCENARIO...`;

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const json = await response.json();
        if (json.status === "success") {
            updateDashboard(json.data, commValid.length);
        } else {
            alert(`Error: ${json.message}`);
        }
    } catch (err) {
        console.error("API error:", err);
        alert("Failed to analyze scenario. Ensure backend server is running.");
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span class="btn-sparkle">✦</span> ANALYZE SCENARIO &amp; GET MOVE`;
    }
}

// Update Dashboard UI with API Results
function updateDashboard(data, commCount) {
    // Stage badge
    const stages = ["PREFLOP", "FLOP", "TURN", "RIVER"];
    const stageName = commCount >= 5 ? stages[3] : commCount >= 4 ? stages[2] : commCount >= 3 ? stages[1] : stages[0];
    document.getElementById("gameStageBadge").innerText = stageName;

    // Action & Bet
    const actionText = data.recommended_action.replace("_", " ").toUpperCase();
    const heroBox = document.getElementById("heroBox");
    
    // Set theme class
    heroBox.className = "hero-decision-box ";
    if (data.action_type === "raise") heroBox.classList.add("action-raise");
    else if (data.action_type === "call") heroBox.classList.add("action-call");
    else if (data.action_type === "check") heroBox.classList.add("action-check");
    else heroBox.classList.add("action-fold");

    document.getElementById("recommendedActionText").innerText = actionText;
    
    if (data.action_type === "fold" || data.action_type === "check") {
        document.getElementById("betAmountContainer").style.display = "none";
    } else {
        document.getElementById("betAmountContainer").style.display = "inline-flex";
        document.getElementById("recommendedBetText").innerText = `$${data.recommended_bet_amount} chips`;
    }

    const valEst = data.state_value_estimate >= 0 ? `+${data.state_value_estimate.toFixed(2)}` : data.state_value_estimate.toFixed(2);
    document.getElementById("valueEstimate").innerText = `EV: ${valEst} BB`;

    // Probability bars
    renderProbabilityBars(data.action_probabilities, data.recommended_action);

    // Statistical metrics
    const analysis = data.hand_analysis;
    document.getElementById("winRateVal").innerText = `${(analysis.expected_win_rate * 100).toFixed(1)}%`;
    document.getElementById("winRate2pVal").innerText = `${(analysis.expected_2_player_win_rate * 100).toFixed(1)}%`;
    document.getElementById("percentileVal").innerText = `${analysis.percentile_rank.toFixed(1)}th`;
    document.getElementById("sklanskyVal").innerText = `Group ${analysis.sklansky_group}`;
    document.getElementById("kellyVal").innerText = `${(analysis.ideal_kelly_max * 100).toFixed(1)}%`;
    document.getElementById("potOddsVal").innerText = `${(analysis.pot_odds * 100).toFixed(1)}%`;
}

// Render Probability Bars
function renderProbabilityBars(probDict, recommendedAction) {
    const list = document.getElementById("probsList");
    list.innerHTML = "";

    const labels = {
        fold: "Fold",
        check_call: "Check / Call",
        raise_small: "Raise Small (Min/0.5x Pot)",
        raise_med: "Raise Medium (1.0x Pot)",
        raise_allin: "Raise All-In"
    };

    Object.entries(probDict).forEach(([key, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const row = document.createElement("div");
        row.className = "prob-row";

        const isDominant = key === recommendedAction;

        row.innerHTML = `
            <div class="prob-info">
                <span class="prob-name">${labels[key] || key}</span>
                <span class="prob-val">${pct}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill ${isDominant ? 'fill-dominant' : ''}" style="width: ${pct}%;"></div>
            </div>
        `;
        list.appendChild(row);
    });
}
