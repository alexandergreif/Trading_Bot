# Bitcoin Futures Trading Bot

Ein lokal laufender, vollständig autonomer Bot, der nach der Liquidity-Sweep-Order-Block-Strategie handelt, die Positionsgröße strikt am Kontorisiko (≤ 1% je Trade) ausrichtet und alle Trades, KPIs & Logs lokal persistiert.

## Systemarchitektur

```
┌────────────┐     WebSocket      ┌──────────────┐
│ Bitunix    │──── Depth Stream ─▶│ LSOB-Detector│
│ Futures API│                   └──────┬───────┘
│  (REST/WS) │◀── Orders / Bal ──┐       │Signal
└────────────┘                  │       ▼
                                │ ┌──────────────┐  SQLite   ┌────────────┐
                                └▶│Position-Mgr  ├──────────▶│ trades.db  │
                                  └──────────────┘           └────────────┘
                                                 ▲
                                                 │ (read-only)
                                   ┌─────────────┴─────────────┐
                                   │ Streamlit Dashboard (UI)  │
                                   └───────────────────────────┘
```

## Quickstart

1. **Repository klonen und ins Verzeichnis wechseln**
   ```bash
   git clone <REPO_URL>
   cd trading_bot
   ```

2. **Virtuelle Umgebung erstellen und aktivieren**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unter Windows: venv\Scripts\activate
   ```

3. **Abhängigkeiten installieren**
   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurationsdatei anlegen**
   ```bash
   cp config.ini.example config.ini
   # Trage deine API-Daten und Einstellungen in config.ini ein
   ```

5. **Bot initialisieren (Konfiguration & Datenbank)**
   ```bash
   python -m trading_bot.cli.main init
   ```

6. **Bot starten (Live-Trading)**
   ```bash
   python -m trading_bot.cli.main run
   ```

7. **Dashboard öffnen (optional)**
   ```bash
   python -m trading_bot.cli.main dashboard
   ```

---

## Installation

```bash
# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Konfiguration

1. Erstellen Sie eine Konfigurationsdatei `config.ini` im Hauptverzeichnis:

```ini
[bitunix]
api_key = IHRE_API_KEY
api_secret = IHR_API_SECRET
testnet = false

[trading]
risk_per_trade = 0.01  # 1% Risiko pro Trade
```

## Verwendung

```bash
# Bot initialisieren (Konfiguration & Datenbank)
python -m trading_bot.cli.main init

# Live-Trading starten
python -m trading_bot.cli.main run

# Dashboard starten
python -m trading_bot.cli.main dashboard
```

## Projektstruktur

```
trading_bot/
├── exchange/       # Bitunix API Client
├── strategy/       # LSOB Strategy
├── trading/        # Position & Risk Management
├── data/           # SQLite Interface
├── ui/             # Streamlit Dashboard
├── backtest/       # Backtesting Engine
├── cli/            # Command Line Interface
└── tests/          # Unit & Integration Tests
```

## Definition of Done

- Live-Bot läuft ≥ 72h ohne Absturz
- Win-Rate-Tracker aktualisiert sich, DB enthält Trades
- Dashboard zeigt Echt-Zeit, keine Null-/NaN-Werte
- README: Install → Config → Start in < 5 Minuten
- pytest > 90% Line-Coverage Kernmodule
