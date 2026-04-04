"""
Market Event Metadata Catalog
─────────────────────────────
Curated database of macro-economic, geopolitical, and policy events that
affect Indian (Nifty 50) and global markets.  Each event has a date range,
category, description, and source reference so users can do event-driven
analysis, e.g. "how does the market behave during wars?"

Sources (all publicly available / reputable):
  • Reserve Bank of India (rbi.org.in) — monetary policy dates & rate changes
  • Ministry of Finance, India (indiabudget.gov.in) — Union Budget dates
  • US Federal Reserve (federalreserve.gov) — FOMC decisions
  • Election Commission of India (eci.gov.in) — general election dates
  • OPEC (opec.org) — oil supply decisions
  • IMF / World Bank crisis timelines
  • Reuters, Bloomberg event archives (dates cross-verified)

Coverage: Sep 2007 – Mar 2026 (matching Nifty 50 data range)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Data Model ────────────────────────────────────────────────────────────

@dataclass
class MarketEvent:
    """A single market event with metadata."""
    name: str
    category: str           # war, oil_shock, budget, rbi_policy, fed_meeting, election, ...
    start_date: date
    end_date: Optional[date] = None   # None → single-day event
    description: str = ""
    source: str = ""
    region: str = "global"  # india | global | both
    impact: str = "medium"  # low | medium | high | extreme
    tags: list[str] = field(default_factory=list)

    @property
    def date_range(self) -> tuple[date, date]:
        return (self.start_date, self.end_date or self.start_date)


# ── Category Definitions ─────────────────────────────────────────────────

CATEGORIES = {
    "war":              "Armed conflicts and military operations",
    "geopolitical":     "Geopolitical tensions, sanctions, diplomatic crises",
    "terror_attack":    "Major terrorist attacks impacting markets",
    "oil_shock":        "Major oil price shocks (spikes or crashes)",
    "financial_crisis": "Systemic financial crises / market crashes",
    "pandemic":         "Pandemics and major health emergencies",
    "india_budget":     "India Union Budget presentation dates",
    "rbi_policy":       "RBI monetary policy rate change decisions",
    "fed_meeting":      "US Federal Reserve major rate decisions",
    "india_election":   "India general / state elections with market impact",
    "us_election":      "US presidential elections",
    "policy_reform":    "Major government policy reforms (India)",
    "trade_war":        "International trade disputes and tariff actions",
    "natural_disaster": "Earthquakes, floods, cyclones with economic impact",
    "corporate_crisis": "Major corporate defaults / fraud with systemic impact",
}


# ── Curated Event Catalog ────────────────────────────────────────────────
# Each event is cross-referenced with publicly available dates from the
# sources listed in the module docstring.  Date ranges represent the
# period of acute market impact, not the whole historical episode.

EVENTS: list[MarketEvent] = [

    # ━━━ WARS & ARMED CONFLICTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    MarketEvent(
        name="Russia-Georgia War",
        category="war",
        start_date=date(2008, 8, 8),
        end_date=date(2008, 8, 16),
        description="Five-day war between Russia and Georgia over South Ossetia.",
        source="Reuters; UN ceasefire agreement 2008-08-12",
        region="global", impact="medium",
        tags=["war", "russia", "geopolitical"],
    ),
    MarketEvent(
        name="Libya Civil War & NATO intervention",
        category="war",
        start_date=date(2011, 2, 15),
        end_date=date(2011, 10, 23),
        description="Libyan civil war and NATO military intervention; oil supply disrupted.",
        source="UN Security Council Resolution 1973; Reuters timeline",
        region="global", impact="high",
        tags=["war", "oil", "nato", "middle_east"],
    ),
    MarketEvent(
        name="Syria Civil War escalation",
        category="war",
        start_date=date(2013, 8, 21),
        end_date=date(2013, 9, 14),
        description="Chemical weapons attack in Ghouta; US threatened military strikes before Russia-brokered deal.",
        source="UN Mission Report 2013; Reuters",
        region="global", impact="medium",
        tags=["war", "syria", "middle_east", "oil"],
    ),
    MarketEvent(
        name="India-Pakistan Pulwama-Balakot Crisis",
        category="war",
        start_date=date(2019, 2, 14),
        end_date=date(2019, 3, 1),
        description="Pulwama terror attack (Feb 14) → Indian Balakot air strikes (Feb 26) → aerial dogfight & de-escalation.",
        source="Ministry of External Affairs, India; Reuters",
        region="india", impact="high",
        tags=["war", "india", "pakistan", "geopolitical"],
    ),
    MarketEvent(
        name="US-Iran tensions (Soleimani)",
        category="war",
        start_date=date(2020, 1, 3),
        end_date=date(2020, 1, 8),
        description="US drone strike killed Iranian General Soleimani; Iran retaliated with missile strikes on US bases in Iraq.",
        source="US DoD statement 2020-01-03; Reuters",
        region="global", impact="high",
        tags=["war", "us", "iran", "oil", "middle_east"],
    ),
    MarketEvent(
        name="Russia-Ukraine War",
        category="war",
        start_date=date(2022, 2, 24),
        end_date=date(2022, 12, 31),
        description="Russia full-scale invasion of Ukraine. Energy crisis, commodity spikes, sanctions.",
        source="UN General Assembly; Reuters; Bloomberg",
        region="global", impact="extreme",
        tags=["war", "russia", "ukraine", "oil", "energy", "sanctions", "commodity"],
    ),
    MarketEvent(
        name="Israel-Hamas War (Gaza)",
        category="war",
        start_date=date(2023, 10, 7),
        end_date=date(2024, 6, 30),
        description="Hamas attack on Israel (Oct 7); Israeli military operation in Gaza; regional tensions (Houthis, Hezbollah).",
        source="IDF statements; UN OCHA reports; Reuters",
        region="global", impact="high",
        tags=["war", "israel", "hamas", "middle_east", "oil"],
    ),

    # ━━━ GEOPOLITICAL TENSIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    MarketEvent(
        name="India-China Doklam Standoff",
        category="geopolitical",
        start_date=date(2017, 6, 16),
        end_date=date(2017, 8, 28),
        description="73-day military standoff between India and China at Doklam plateau near Bhutan tri-junction.",
        source="Ministry of External Affairs, India; Xinhua",
        region="india", impact="medium",
        tags=["india", "china", "border", "geopolitical"],
    ),
    MarketEvent(
        name="India-China Galwan Valley Clash",
        category="geopolitical",
        start_date=date(2020, 6, 15),
        end_date=date(2020, 7, 15),
        description="Deadly clash between Indian and Chinese troops at Galwan Valley, Ladakh. 20 Indian soldiers killed.",
        source="Ministry of Defence, India; Reuters",
        region="india", impact="high",
        tags=["india", "china", "border", "geopolitical"],
    ),
    MarketEvent(
        name="North Korea Nuclear/Missile Tests",
        category="geopolitical",
        start_date=date(2017, 7, 4),
        end_date=date(2017, 11, 29),
        description="Escalating North Korea ICBM tests and nuclear test (Sep 3); 'fire and fury' rhetoric.",
        source="UN Security Council; IAEA; Reuters",
        region="global", impact="medium",
        tags=["north_korea", "nuclear", "geopolitical"],
    ),

    # ━━━ TERROR ATTACKS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    MarketEvent(
        name="Mumbai 26/11 Terror Attacks",
        category="terror_attack",
        start_date=date(2008, 11, 26),
        end_date=date(2008, 11, 29),
        description="Coordinated terrorist attacks across Mumbai (Taj Hotel, CST station, etc.). 166 killed.",
        source="National Investigation Agency (NIA), India; Reuters",
        region="india", impact="extreme",
        tags=["terror", "india", "mumbai"],
    ),
    MarketEvent(
        name="Pathankot Air Force Station Attack",
        category="terror_attack",
        start_date=date(2016, 1, 2),
        end_date=date(2016, 1, 4),
        description="Terror attack on Indian Air Force Station Pathankot.",
        source="Ministry of Defence, India; NDTV",
        region="india", impact="medium",
        tags=["terror", "india"],
    ),
    MarketEvent(
        name="Uri Army Base Attack & Surgical Strikes",
        category="terror_attack",
        start_date=date(2016, 9, 18),
        end_date=date(2016, 9, 29),
        description="Terror attack on Uri army base (Sep 18); Indian surgical strikes across LoC (Sep 29).",
        source="Indian Army; DGMO press briefing 2016-09-29",
        region="india", impact="high",
        tags=["terror", "india", "pakistan", "surgical_strike"],
    ),

    # ━━━ OIL SHOCKS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    MarketEvent(
        name="2008 Oil Price Spike ($147)",
        category="oil_shock",
        start_date=date(2008, 1, 2),
        end_date=date(2008, 7, 14),
        description="Crude oil surged to all-time high $147.27/barrel (Jul 11) driven by speculation and demand.",
        source="NYMEX; EIA data; Reuters",
        region="global", impact="extreme",
        tags=["oil", "commodity", "inflation"],
    ),
    MarketEvent(
        name="2008 Oil Price Crash",
        category="oil_shock",
        start_date=date(2008, 7, 15),
        end_date=date(2008, 12, 23),
        description="Oil crashed from $147 to $33 as GFC destroyed demand.",
        source="NYMEX; EIA; Reuters",
        region="global", impact="extreme",
        tags=["oil", "commodity", "financial_crisis"],
    ),
    MarketEvent(
        name="2014-16 Oil Price Crash",
        category="oil_shock",
        start_date=date(2014, 6, 20),
        end_date=date(2016, 2, 11),
        description="Oil fell from $115 to $26 due to US shale supply glut and OPEC refusing to cut production.",
        source="OPEC; EIA; Bloomberg",
        region="global", impact="high",
        tags=["oil", "commodity", "opec", "shale"],
    ),
    MarketEvent(
        name="Saudi Aramco Drone Attack",
        category="oil_shock",
        start_date=date(2019, 9, 14),
        end_date=date(2019, 9, 20),
        description="Drone and missile attack on Saudi Aramco facilities at Abqaiq. Oil spiked ~15% in single day.",
        source="Saudi Aramco; Reuters; IEA",
        region="global", impact="high",
        tags=["oil", "saudi", "geopolitical", "supply_disruption"],
    ),
    MarketEvent(
        name="2020 Oil Price War (Saudi-Russia)",
        category="oil_shock",
        start_date=date(2020, 3, 6),
        end_date=date(2020, 4, 20),
        description="Saudi-Russia price war collapsed oil; WTI went negative (-$37.63) on Apr 20.",
        source="OPEC; CME/NYMEX; Reuters",
        region="global", impact="extreme",
        tags=["oil", "opec", "russia", "saudi", "commodity"],
    ),
    MarketEvent(
        name="2022 Energy Crisis (Ukraine War)",
        category="oil_shock",
        start_date=date(2022, 2, 24),
        end_date=date(2022, 6, 14),
        description="Oil spiked above $120 and European gas prices surged following Russia-Ukraine war and sanctions.",
        source="IEA; EIA; Bloomberg",
        region="global", impact="extreme",
        tags=["oil", "energy", "gas", "russia", "ukraine", "sanctions"],
    ),

    # ━━━ FINANCIAL CRISES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    MarketEvent(
        name="Global Financial Crisis (GFC / Lehman)",
        category="financial_crisis",
        start_date=date(2008, 9, 15),
        end_date=date(2009, 3, 9),
        description="Lehman Brothers bankruptcy triggered global credit freeze. Nifty fell ~60% from peak. "
                    "Bear Stearns rescued Mar 2008; AIG bailout Sep 2008; TARP Oct 2008.",
        source="US Treasury; Federal Reserve; IMF WEO Oct 2008; NSE historical data",
        region="global", impact="extreme",
        tags=["financial_crisis", "lehman", "banking", "credit_crisis"],
    ),
    MarketEvent(
        name="European Sovereign Debt Crisis",
        category="financial_crisis",
        start_date=date(2010, 4, 23),
        end_date=date(2012, 7, 26),
        description="Greece requested bailout (Apr 2010). Spread to Ireland, Portugal, Spain, Italy. "
                    "ECB's Draghi 'whatever it takes' (Jul 26, 2012) marked the turning point.",
        source="ECB; IMF; Eurostat; Reuters",
        region="global", impact="high",
        tags=["financial_crisis", "europe", "sovereign_debt", "greece"],
    ),
    MarketEvent(
        name="US Debt Ceiling Crisis & S&P Downgrade",
        category="financial_crisis",
        start_date=date(2011, 7, 22),
        end_date=date(2011, 8, 8),
        description="S&P downgraded US from AAA to AA+ (Aug 5) amid debt ceiling standoff. Global selloff.",
        source="S&P Global Ratings; US Treasury; Reuters",
        region="global", impact="high",
        tags=["financial_crisis", "us", "debt_ceiling", "credit_downgrade"],
    ),
    MarketEvent(
        name="Taper Tantrum",
        category="financial_crisis",
        start_date=date(2013, 5, 22),
        end_date=date(2013, 9, 18),
        description="Bernanke hinted at tapering QE (May 22). Massive EM selloff; Indian rupee crashed to ₹68.85/$. "
                    "RBI's Rajan appointed Sep 4, Nifty recovered after Fed delayed taper (Sep 18).",
        source="Federal Reserve; RBI; Bloomberg",
        region="both", impact="high",
        tags=["financial_crisis", "taper", "fed", "rupee", "em_selloff"],
    ),
    MarketEvent(
        name="China Stock Market Crash",
        category="financial_crisis",
        start_date=date(2015, 6, 12),
        end_date=date(2016, 2, 12),
        description="Shanghai Composite lost ~45%. 'Black Monday' Aug 24 triggered global selloff. "
                    "Yuan devaluation Aug 11. Circuit breakers introduced Jan 2016 then suspended.",
        source="Shanghai Stock Exchange; PBoC; Bloomberg; Reuters",
        region="global", impact="high",
        tags=["financial_crisis", "china", "yuan", "devaluation"],
    ),
    MarketEvent(
        name="India IL&FS / NBFC Crisis",
        category="financial_crisis",
        start_date=date(2018, 9, 1),
        end_date=date(2019, 6, 30),
        description="IL&FS default (Sep 2018) triggered cascading NBFC liquidity crisis. "
                    "DHFL, Yes Bank stressed. Credit freeze for NBFCs/HFCs.",
        source="RBI Financial Stability Report Dec 2018; SEBI; Reuters",
        region="india", impact="extreme",
        tags=["financial_crisis", "india", "nbfc", "ilfs", "credit_crisis"],
    ),
    MarketEvent(
        name="COVID-19 Market Crash",
        category="financial_crisis",
        start_date=date(2020, 2, 20),
        end_date=date(2020, 3, 23),
        description="Fastest bear market in history. Nifty fell ~38% from peak (12,430 → 7,511). "
                    "US circuit breakers triggered 4 times. Global lockdowns.",
        source="NSE; NYSE; WHO; IMF; Bloomberg",
        region="global", impact="extreme",
        tags=["financial_crisis", "covid", "pandemic", "lockdown"],
    ),
    MarketEvent(
        name="US Regional Banking Crisis (SVB)",
        category="financial_crisis",
        start_date=date(2023, 3, 8),
        end_date=date(2023, 5, 1),
        description="Silicon Valley Bank collapsed (Mar 10), Signature Bank (Mar 12), First Republic (May 1). "
                    "Fed created emergency lending facility (BTFP).",
        source="FDIC; Federal Reserve; Bloomberg; Reuters",
        region="global", impact="high",
        tags=["financial_crisis", "banking", "svb", "us"],
    ),
    MarketEvent(
        name="Adani-Hindenburg Short Report",
        category="corporate_crisis",
        start_date=date(2023, 1, 24),
        end_date=date(2023, 3, 2),
        description="Hindenburg Research published short-seller report on Adani Group. "
                    "Adani stocks lost ~$150B; Nifty dragged down. SEBI investigation followed.",
        source="Hindenburg Research; SEBI; NSE; Bloomberg",
        region="india", impact="high",
        tags=["corporate_crisis", "india", "adani", "fraud_allegation"],
    ),

    # ━━━ PANDEMICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    MarketEvent(
        name="COVID-19 Pandemic Declared",
        category="pandemic",
        start_date=date(2020, 1, 30),
        end_date=date(2020, 3, 23),
        description="WHO declared global health emergency (Jan 30) then pandemic (Mar 11). "
                    "India's first case Jan 30. Markets in freefall until Mar 23 bottom.",
        source="WHO; MoHFW India; Johns Hopkins CSSE",
        region="global", impact="extreme",
        tags=["pandemic", "covid", "health", "lockdown"],
    ),
    MarketEvent(
        name="India COVID Lockdown",
        category="pandemic",
        start_date=date(2020, 3, 24),
        end_date=date(2020, 6, 1),
        description="PM Modi announced nationwide lockdown (Mar 24). Extended through May. Unlock 1.0 from Jun 1.",
        source="Ministry of Home Affairs, India; PIB",
        region="india", impact="extreme",
        tags=["pandemic", "covid", "lockdown", "india"],
    ),
    MarketEvent(
        name="India COVID Second Wave",
        category="pandemic",
        start_date=date(2021, 3, 15),
        end_date=date(2021, 6, 15),
        description="Devastating Delta variant wave. Oxygen shortages, healthcare system overwhelmed. "
                    "Daily cases peaked >400K in May.",
        source="MoHFW India; WHO India; Reuters",
        region="india", impact="high",
        tags=["pandemic", "covid", "delta", "india"],
    ),

    # ━━━ INDIA UNION BUDGETS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Source: indiabudget.gov.in; PRS Legislative Research

    MarketEvent(
        name="Union Budget 2008-09",
        category="india_budget",
        start_date=date(2008, 2, 29),
        description="FM P. Chidambaram. Farm loan waiver ₹71,680 crore. Higher fiscal deficit.",
        source="indiabudget.gov.in",
        region="india", impact="high",
        tags=["budget", "india", "fiscal_policy"],
    ),
    MarketEvent(
        name="Interim Budget 2009-10",
        category="india_budget",
        start_date=date(2009, 2, 16),
        description="FM Pranab Mukherjee. Vote-on-account before elections.",
        source="indiabudget.gov.in",
        region="india", impact="low",
        tags=["budget", "india"],
    ),
    MarketEvent(
        name="Union Budget 2009-10 (Full)",
        category="india_budget",
        start_date=date(2009, 7, 6),
        description="FM Pranab Mukherjee. Post-election budget. Disinvestment push.",
        source="indiabudget.gov.in",
        region="india", impact="medium",
        tags=["budget", "india", "fiscal_policy"],
    ),
    MarketEvent(
        name="Union Budget 2010-11",
        category="india_budget",
        start_date=date(2010, 2, 26),
        description="FM Pranab Mukherjee. Partial GST rollout plan; fiscal consolidation.",
        source="indiabudget.gov.in",
        region="india", impact="medium",
        tags=["budget", "india"],
    ),
    MarketEvent(
        name="Union Budget 2011-12",
        category="india_budget",
        start_date=date(2011, 2, 28),
        description="FM Pranab Mukherjee. DTC (Direct Taxes Code) proposal.",
        source="indiabudget.gov.in",
        region="india", impact="medium",
        tags=["budget", "india"],
    ),
    MarketEvent(
        name="Union Budget 2012-13",
        category="india_budget",
        start_date=date(2012, 3, 16),
        description="FM Pranab Mukherjee. GAAR (anti-avoidance) proposal rattled FIIs.",
        source="indiabudget.gov.in",
        region="india", impact="high",
        tags=["budget", "india", "gaar", "fii"],
    ),
    MarketEvent(
        name="Union Budget 2013-14",
        category="india_budget",
        start_date=date(2013, 2, 28),
        description="FM P. Chidambaram. Fiscal consolidation roadmap; investment tax breaks.",
        source="indiabudget.gov.in",
        region="india", impact="medium",
        tags=["budget", "india"],
    ),
    MarketEvent(
        name="Interim Budget 2014-15",
        category="india_budget",
        start_date=date(2014, 2, 17),
        description="FM P. Chidambaram. Vote-on-account before 2014 elections.",
        source="indiabudget.gov.in",
        region="india", impact="low",
        tags=["budget", "india"],
    ),
    MarketEvent(
        name="Union Budget 2014-15 (Full – Modi Govt 1st)",
        category="india_budget",
        start_date=date(2014, 7, 10),
        description="FM Arun Jaitley. First Modi govt budget. FDI reforms, Make in India vision.",
        source="indiabudget.gov.in",
        region="india", impact="high",
        tags=["budget", "india", "reform", "fdi"],
    ),
    MarketEvent(
        name="Union Budget 2015-16",
        category="india_budget",
        start_date=date(2015, 2, 28),
        description="FM Arun Jaitley. Corporate tax cut roadmap (30%→25% over 4 years). GST push.",
        source="indiabudget.gov.in",
        region="india", impact="high",
        tags=["budget", "india", "tax_reform"],
    ),
    MarketEvent(
        name="Union Budget 2016-17",
        category="india_budget",
        start_date=date(2016, 2, 29),
        description="FM Arun Jaitley. Fiscal discipline; infrastructure push; LTCG holding period changed.",
        source="indiabudget.gov.in",
        region="india", impact="medium",
        tags=["budget", "india"],
    ),
    MarketEvent(
        name="Union Budget 2017-18 (moved to Feb 1)",
        category="india_budget",
        start_date=date(2017, 2, 1),
        description="FM Arun Jaitley. First budget on Feb 1 (moved from end-Feb). Political funding via electoral bonds.",
        source="indiabudget.gov.in",
        region="india", impact="medium",
        tags=["budget", "india", "electoral_bonds"],
    ),
    MarketEvent(
        name="Union Budget 2018-19",
        category="india_budget",
        start_date=date(2018, 2, 1),
        description="FM Arun Jaitley. LTCG tax of 10% reintroduced on equities. Market fell ~2% on budget day.",
        source="indiabudget.gov.in",
        region="india", impact="high",
        tags=["budget", "india", "ltcg", "tax"],
    ),
    MarketEvent(
        name="Interim Budget 2019-20",
        category="india_budget",
        start_date=date(2019, 2, 1),
        description="FM Piyush Goyal. PM-KISAN scheme; tax rebate up to ₹5L income.",
        source="indiabudget.gov.in",
        region="india", impact="medium",
        tags=["budget", "india"],
    ),
    MarketEvent(
        name="Union Budget 2019-20 (Full)",
        category="india_budget",
        start_date=date(2019, 7, 5),
        description="FM Nirmala Sitharaman. Super-rich surcharge on FPIs rattled markets. Later rolled back.",
        source="indiabudget.gov.in",
        region="india", impact="high",
        tags=["budget", "india", "fpi", "surcharge"],
    ),
    MarketEvent(
        name="Union Budget 2020-21",
        category="india_budget",
        start_date=date(2020, 2, 1),
        description="FM Nirmala Sitharaman. New optional tax regime; DDT abolished.",
        source="indiabudget.gov.in",
        region="india", impact="medium",
        tags=["budget", "india", "tax_reform"],
    ),
    MarketEvent(
        name="Union Budget 2021-22",
        category="india_budget",
        start_date=date(2021, 2, 1),
        description="FM Nirmala Sitharaman. Massive infra push, BFSI recapitalisation. Nifty rallied ~5% on day.",
        source="indiabudget.gov.in",
        region="india", impact="high",
        tags=["budget", "india", "infrastructure", "banking"],
    ),
    MarketEvent(
        name="Union Budget 2022-23",
        category="india_budget",
        start_date=date(2022, 2, 1),
        description="FM Nirmala Sitharaman. Digital rupee announced; capex ₹7.5L crore.",
        source="indiabudget.gov.in",
        region="india", impact="medium",
        tags=["budget", "india", "digital_rupee", "capex"],
    ),
    MarketEvent(
        name="Union Budget 2023-24",
        category="india_budget",
        start_date=date(2023, 2, 1),
        description="FM Nirmala Sitharaman. Capex ₹10L crore; new tax regime made default.",
        source="indiabudget.gov.in",
        region="india", impact="medium",
        tags=["budget", "india", "capex"],
    ),
    MarketEvent(
        name="Interim Budget 2024-25",
        category="india_budget",
        start_date=date(2024, 2, 1),
        description="FM Nirmala Sitharaman. Vote-on-account before 2024 elections. No major tax changes.",
        source="indiabudget.gov.in",
        region="india", impact="low",
        tags=["budget", "india"],
    ),
    MarketEvent(
        name="Union Budget 2024-25 (Full)",
        category="india_budget",
        start_date=date(2024, 7, 23),
        description="FM Nirmala Sitharaman. LTCG raised to 12.5%; STT on F&O doubled. Indexation removed from property.",
        source="indiabudget.gov.in",
        region="india", impact="high",
        tags=["budget", "india", "ltcg", "stt", "tax"],
    ),
    MarketEvent(
        name="Union Budget 2025-26",
        category="india_budget",
        start_date=date(2025, 2, 1),
        description="FM Nirmala Sitharaman. Income tax exemption raised to ₹12L; focus on middle class, consumption boost.",
        source="indiabudget.gov.in",
        region="india", impact="high",
        tags=["budget", "india", "tax_reform", "consumption"],
    ),

    # ━━━ RBI MONETARY POLICY (major rate changes) ━━━━━━━━━━━━━━━━━━━━━━━
    # Source: RBI Monetary Policy Statements (rbi.org.in)
    # Only rate change decisions are included, not status-quo meetings.

    MarketEvent(
        name="RBI Emergency Rate Cut (GFC)",
        category="rbi_policy",
        start_date=date(2008, 10, 20),
        description="RBI cut repo rate by 100 bps to 7.5% in emergency action during GFC.",
        source="RBI Monetary Policy Statement Oct 2008",
        region="india", impact="high",
        tags=["rbi", "rate_cut", "monetary_policy"],
    ),
    MarketEvent(
        name="RBI Rate Cuts 2009 (series)",
        category="rbi_policy",
        start_date=date(2009, 1, 5),
        end_date=date(2009, 4, 21),
        description="RBI cut repo from 6.5% to 4.75% in multiple steps to support growth post-GFC.",
        source="RBI",
        region="india", impact="high",
        tags=["rbi", "rate_cut", "monetary_policy"],
    ),
    MarketEvent(
        name="RBI Rate Hike Cycle 2010-11",
        category="rbi_policy",
        start_date=date(2010, 3, 19),
        end_date=date(2011, 10, 25),
        description="RBI hiked repo 13 times from 4.75% to 8.5% to fight inflation (incl. food/oil).",
        source="RBI",
        region="india", impact="high",
        tags=["rbi", "rate_hike", "inflation", "monetary_policy"],
    ),
    MarketEvent(
        name="RBI Rajan Rate Cut Cycle",
        category="rbi_policy",
        start_date=date(2015, 1, 15),
        end_date=date(2016, 4, 5),
        description="Governor Rajan cut repo from 8% to 6.5% as inflation moderated.",
        source="RBI Monetary Policy Statements 2015-2016",
        region="india", impact="high",
        tags=["rbi", "rate_cut", "rajan", "monetary_policy"],
    ),
    MarketEvent(
        name="RBI COVID Emergency Rate Cut",
        category="rbi_policy",
        start_date=date(2020, 3, 27),
        description="RBI cut repo by 75 bps to 4.4% in emergency off-cycle action; CRR cut by 100 bps.",
        source="RBI Press Release 2020-03-27",
        region="india", impact="high",
        tags=["rbi", "rate_cut", "covid", "monetary_policy"],
    ),
    MarketEvent(
        name="RBI Rate Hike Cycle 2022-23",
        category="rbi_policy",
        start_date=date(2022, 5, 4),
        end_date=date(2023, 2, 8),
        description="RBI hiked repo from 4.0% to 6.5% (250 bps) to combat post-COVID inflation.",
        source="RBI MPC Statements 2022-2023",
        region="india", impact="high",
        tags=["rbi", "rate_hike", "inflation", "monetary_policy"],
    ),
    MarketEvent(
        name="RBI Rate Cut Cycle 2025",
        category="rbi_policy",
        start_date=date(2025, 2, 7),
        end_date=date(2025, 4, 9),
        description="RBI began easing — cut repo by 25 bps to 6.25% (Feb), then another 25 bps to 6.0% (Apr).",
        source="RBI MPC Statement Feb 2025; Apr 2025",
        region="india", impact="high",
        tags=["rbi", "rate_cut", "monetary_policy"],
    ),

    # ━━━ US FEDERAL RESERVE (major decisions) ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Source: federalreserve.gov FOMC decisions

    MarketEvent(
        name="Fed Emergency Cuts (GFC)",
        category="fed_meeting",
        start_date=date(2007, 9, 18),
        end_date=date(2008, 12, 16),
        description="Fed cut FFR from 5.25% to 0-0.25% in series of emergency cuts during GFC. "
                    "Including 75 bps inter-meeting cut (Jan 22, 2008).",
        source="FOMC Statements; Federal Reserve",
        region="global", impact="extreme",
        tags=["fed", "rate_cut", "gfc", "monetary_policy"],
    ),
    MarketEvent(
        name="Fed QE1 Launch",
        category="fed_meeting",
        start_date=date(2008, 11, 25),
        description="Fed announced purchase of $600B in MBS — first quantitative easing program.",
        source="Federal Reserve Press Release Nov 2008",
        region="global", impact="extreme",
        tags=["fed", "qe", "monetary_policy"],
    ),
    MarketEvent(
        name="Fed QE2 Announcement",
        category="fed_meeting",
        start_date=date(2010, 11, 3),
        description="Fed announced $600B Treasury purchase program (QE2).",
        source="FOMC Statement Nov 2010",
        region="global", impact="high",
        tags=["fed", "qe", "monetary_policy"],
    ),
    MarketEvent(
        name="Fed QE3 (Open-ended)",
        category="fed_meeting",
        start_date=date(2012, 9, 13),
        description="Fed announced open-ended QE3 — $40B/month MBS (later expanded to $85B).",
        source="FOMC Statement Sep 2012",
        region="global", impact="high",
        tags=["fed", "qe", "monetary_policy"],
    ),
    MarketEvent(
        name="Fed Taper Hint (Bernanke)",
        category="fed_meeting",
        start_date=date(2013, 5, 22),
        description="In Congressional testimony, Bernanke said Fed 'could begin tapering'. Triggered Taper Tantrum.",
        source="Federal Reserve; Congressional testimony transcript",
        region="global", impact="extreme",
        tags=["fed", "taper", "monetary_policy"],
    ),
    MarketEvent(
        name="Fed First Rate Hike (post-GFC)",
        category="fed_meeting",
        start_date=date(2015, 12, 16),
        description="Fed raised FFR for first time since 2006 — from 0-0.25% to 0.25-0.50%.",
        source="FOMC Statement Dec 2015",
        region="global", impact="high",
        tags=["fed", "rate_hike", "monetary_policy"],
    ),
    MarketEvent(
        name="Fed COVID Emergency Cut to Zero",
        category="fed_meeting",
        start_date=date(2020, 3, 15),
        description="Fed cut FFR to 0-0.25% in emergency Sunday action; announced unlimited QE.",
        source="FOMC Emergency Statement Mar 2020",
        region="global", impact="extreme",
        tags=["fed", "rate_cut", "covid", "qe", "monetary_policy"],
    ),
    MarketEvent(
        name="Fed Rate Hike Cycle 2022-23",
        category="fed_meeting",
        start_date=date(2022, 3, 16),
        end_date=date(2023, 7, 26),
        description="Fed hiked from 0% to 5.25-5.50% — fastest tightening in 40 years. "
                    "Includes four successive 75 bps hikes (Jun-Nov 2022).",
        source="FOMC Statements 2022-2023",
        region="global", impact="extreme",
        tags=["fed", "rate_hike", "inflation", "monetary_policy"],
    ),
    MarketEvent(
        name="Fed Rate Cut Cycle Begins (Sep 2024)",
        category="fed_meeting",
        start_date=date(2024, 9, 18),
        end_date=date(2024, 12, 18),
        description="Fed began easing — 50 bps cut (Sep), then 25 bps each (Nov, Dec). FFR to 4.25-4.50%.",
        source="FOMC Statements Sep-Dec 2024",
        region="global", impact="high",
        tags=["fed", "rate_cut", "monetary_policy"],
    ),

    # ━━━ INDIA ELECTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Source: Election Commission of India (eci.gov.in)

    MarketEvent(
        name="India General Election 2009 (UPA-2)",
        category="india_election",
        start_date=date(2009, 4, 16),
        end_date=date(2009, 5, 16),
        description="5-phase election. UPA won clear mandate. Nifty hit upper circuit (+17%) on results day (May 18).",
        source="ECI; NSE Historical Data",
        region="india", impact="extreme",
        tags=["election", "india", "upa"],
    ),
    MarketEvent(
        name="India General Election 2014 (Modi Wave)",
        category="india_election",
        start_date=date(2014, 4, 7),
        end_date=date(2014, 5, 16),
        description="9-phase election. BJP won historic majority (282 seats). Nifty rallied ~25% in election year.",
        source="ECI; NSE Historical Data",
        region="india", impact="extreme",
        tags=["election", "india", "bjp", "modi"],
    ),
    MarketEvent(
        name="India General Election 2019 (Modi 2.0)",
        category="india_election",
        start_date=date(2019, 4, 11),
        end_date=date(2019, 5, 23),
        description="7-phase election. BJP strengthened majority (303 seats). Nifty gapped up ~3% on results.",
        source="ECI; NSE Historical Data",
        region="india", impact="high",
        tags=["election", "india", "bjp", "modi"],
    ),
    MarketEvent(
        name="India General Election 2024 (Modi 3.0)",
        category="india_election",
        start_date=date(2024, 4, 19),
        end_date=date(2024, 6, 4),
        description="7-phase election. NDA won but BJP lost majority (240 seats, coalition dependent). "
                    "Nifty crashed ~6% on Jun 4 (exit polls had predicted sweep).",
        source="ECI; NSE Historical Data; Reuters",
        region="india", impact="extreme",
        tags=["election", "india", "bjp", "modi", "nda"],
    ),

    # ━━━ US ELECTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    MarketEvent(
        name="US Presidential Election 2008 (Obama)",
        category="us_election",
        start_date=date(2008, 11, 4),
        description="Barack Obama elected 44th President amid GFC.",
        source="FEC; Associated Press",
        region="global", impact="high",
        tags=["election", "us", "obama"],
    ),
    MarketEvent(
        name="US Presidential Election 2012 (Obama 2nd)",
        category="us_election",
        start_date=date(2012, 11, 6),
        description="Obama re-elected. Markets initially dipped on fiscal cliff fears.",
        source="FEC; Associated Press",
        region="global", impact="medium",
        tags=["election", "us", "obama"],
    ),
    MarketEvent(
        name="US Presidential Election 2016 (Trump)",
        category="us_election",
        start_date=date(2016, 11, 8),
        description="Donald Trump elected. Initial futures crash reversed to rally ('Trump trade').",
        source="FEC; Associated Press",
        region="global", impact="high",
        tags=["election", "us", "trump"],
    ),
    MarketEvent(
        name="US Presidential Election 2020 (Biden)",
        category="us_election",
        start_date=date(2020, 11, 3),
        end_date=date(2020, 11, 7),
        description="Joe Biden elected. Prolonged count. Markets rallied on split-government hopes.",
        source="FEC; Associated Press",
        region="global", impact="high",
        tags=["election", "us", "biden"],
    ),
    MarketEvent(
        name="US Presidential Election 2024 (Trump 2.0)",
        category="us_election",
        start_date=date(2024, 11, 5),
        description="Donald Trump won second term. Markets rallied; 'Trump 2.0 trade' in crypto, banks, defense.",
        source="FEC; Associated Press; Reuters",
        region="global", impact="high",
        tags=["election", "us", "trump"],
    ),

    # ━━━ INDIA POLICY REFORMS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    MarketEvent(
        name="FDI Reforms (Retail, Aviation, etc.)",
        category="policy_reform",
        start_date=date(2012, 9, 14),
        description="Government opened FDI in multi-brand retail (51%), aviation (49%), broadcasting. UPA reform push.",
        source="DPIIT; PIB; Reuters",
        region="india", impact="high",
        tags=["reform", "india", "fdi"],
    ),
    MarketEvent(
        name="Demonetisation",
        category="policy_reform",
        start_date=date(2016, 11, 8),
        end_date=date(2017, 3, 31),
        description="PM Modi demonetised ₹500 and ₹1000 notes (86% of currency). "
                    "Cash crunch; GDP growth dipped. Shift to digital payments.",
        source="RBI Annual Report 2016-17; PIB",
        region="india", impact="extreme",
        tags=["reform", "india", "demonetisation", "currency"],
    ),
    MarketEvent(
        name="GST Implementation",
        category="policy_reform",
        start_date=date(2017, 7, 1),
        end_date=date(2017, 9, 30),
        description="India rolled out Goods and Services Tax replacing 17 central/state taxes. "
                    "Biggest tax reform since independence. Initial disruption to manufacturing/trade.",
        source="GST Council; CBIC; PIB",
        region="india", impact="extreme",
        tags=["reform", "india", "gst", "tax"],
    ),
    MarketEvent(
        name="Corporate Tax Cut Surprise",
        category="policy_reform",
        start_date=date(2019, 9, 20),
        description="FM Sitharaman slashed corporate tax from 30% to 22% (new cos 15%) via ordinance. "
                    "Nifty surged ~5.3% — biggest single-day gain in 10 years.",
        source="Ministry of Finance; NSE; PIB",
        region="india", impact="extreme",
        tags=["reform", "india", "tax_cut", "corporate"],
    ),
    MarketEvent(
        name="India COVID Stimulus (Atmanirbhar)",
        category="policy_reform",
        start_date=date(2020, 5, 12),
        end_date=date(2020, 5, 17),
        description="PM announced ₹20 lakh crore stimulus package ('Atmanirbhar Bharat'). "
                    "Included MSME support, liquidity measures, labour reforms.",
        source="PIB; Ministry of Finance; Reuters",
        region="india", impact="high",
        tags=["reform", "india", "stimulus", "covid"],
    ),
    MarketEvent(
        name="India Farm Law Protests",
        category="policy_reform",
        start_date=date(2020, 11, 26),
        end_date=date(2021, 11, 19),
        description="Farmers protested three new farm laws at Delhi borders. "
                    "Laws repealed Nov 19, 2021 before UP elections.",
        source="PIB; Reuters; Supreme Court of India",
        region="india", impact="medium",
        tags=["reform", "india", "agriculture", "protest"],
    ),

    # ━━━ TRADE WARS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    MarketEvent(
        name="US-China Trade War (Phase 1)",
        category="trade_war",
        start_date=date(2018, 3, 22),
        end_date=date(2020, 1, 15),
        description="Trump signed tariff memo (Mar 22, 2018). Escalating tariffs through 2019. "
                    "Phase 1 deal signed Jan 15, 2020.",
        source="USTR; Reuters; Bloomberg",
        region="global", impact="extreme",
        tags=["trade_war", "us", "china", "tariffs"],
    ),
    MarketEvent(
        name="US Steel & Aluminum Tariffs",
        category="trade_war",
        start_date=date(2018, 3, 1),
        end_date=date(2018, 6, 1),
        description="Trump imposed 25% steel / 10% aluminum tariffs on allies (EU, Canada, Mexico, India).",
        source="US Commerce Dept; USTR; Reuters",
        region="global", impact="high",
        tags=["trade_war", "tariffs", "steel", "us"],
    ),
    MarketEvent(
        name="Trump 'Liberation Day' Tariffs 2025",
        category="trade_war",
        start_date=date(2025, 4, 2),
        end_date=date(2025, 4, 9),
        description="Trump announced sweeping reciprocal tariffs: 26% on India, 34% on China, 20% on EU. "
                    "Global markets crashed ~10%. 90-day pause announced Apr 9 for non-China countries.",
        source="White House Executive Order; Reuters; Bloomberg",
        region="global", impact="extreme",
        tags=["trade_war", "tariffs", "us", "india", "china"],
    ),

    # ━━━ NATURAL DISASTERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    MarketEvent(
        name="Japan Earthquake, Tsunami & Fukushima",
        category="natural_disaster",
        start_date=date(2011, 3, 11),
        end_date=date(2011, 3, 18),
        description="9.0 earthquake + tsunami killed ~16,000. Fukushima nuclear meltdown. "
                    "Nikkei fell ~17% in 3 days; global supply chains disrupted.",
        source="Japan Meteorological Agency; IAEA; Reuters",
        region="global", impact="extreme",
        tags=["earthquake", "tsunami", "nuclear", "japan", "supply_chain"],
    ),
    MarketEvent(
        name="Nepal Earthquake",
        category="natural_disaster",
        start_date=date(2015, 4, 25),
        end_date=date(2015, 5, 12),
        description="7.8 magnitude earthquake; 9,000 killed. India sent aid. Limited direct market impact.",
        source="USGS; Reuters",
        region="india", impact="low",
        tags=["earthquake", "nepal", "india"],
    ),
    MarketEvent(
        name="Kerala Floods 2018",
        category="natural_disaster",
        start_date=date(2018, 8, 8),
        end_date=date(2018, 8, 30),
        description="Worst Kerala floods in century. 483 killed, ₹40,000 crore damage. Tourism, agriculture hit.",
        source="India Meteorological Department; Kerala Govt; Reuters",
        region="india", impact="medium",
        tags=["flood", "india", "kerala"],
    ),
    MarketEvent(
        name="Cyclone Amphan",
        category="natural_disaster",
        start_date=date(2020, 5, 20),
        end_date=date(2020, 5, 21),
        description="Super cyclone hit West Bengal & Odisha. $13B damage — costliest cyclone in Bay of Bengal.",
        source="India Meteorological Department; World Bank; Reuters",
        region="india", impact="medium",
        tags=["cyclone", "india", "natural_disaster"],
    ),

    # ━━━ CORPORATE CRISES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    MarketEvent(
        name="Satyam Accounting Fraud",
        category="corporate_crisis",
        start_date=date(2009, 1, 7),
        end_date=date(2009, 1, 12),
        description="Satyam Computer chairman Ramalinga Raju confessed to ₹7,136 crore accounting fraud. "
                    "Stock crashed 78% in a day. India's Enron moment.",
        source="SEBI; CBI; NSE; Reuters",
        region="india", impact="extreme",
        tags=["fraud", "india", "corporate", "satyam"],
    ),
    MarketEvent(
        name="Yes Bank Crisis & Moratorium",
        category="corporate_crisis",
        start_date=date(2020, 3, 5),
        end_date=date(2020, 3, 18),
        description="RBI imposed moratorium on Yes Bank (Mar 5); restructured with SBI-led consortium. "
                    "Depositors' withdrawals capped at ₹50,000.",
        source="RBI Order 2020-03-05; SBI; Reuters",
        region="india", impact="high",
        tags=["banking_crisis", "india", "yes_bank"],
    ),
    MarketEvent(
        name="Archegos Capital Collapse",
        category="corporate_crisis",
        start_date=date(2021, 3, 26),
        end_date=date(2021, 3, 29),
        description="Archegos Capital family office margin call. $20B+ losses at Credit Suisse, Nomura. "
                    "Fire sale of $30B in block trades.",
        source="SEC Filing; Bloomberg; Credit Suisse annual report 2021",
        region="global", impact="high",
        tags=["corporate_crisis", "hedge_fund", "margin_call"],
    ),
    MarketEvent(
        name="Credit Suisse Collapse",
        category="corporate_crisis",
        start_date=date(2023, 3, 15),
        end_date=date(2023, 3, 19),
        description="Credit Suisse shares crashed after Saudi investor ruled out more capital. "
                    "Emergency UBS takeover at CHF 3B ($3.25B) brokered by Swiss govt.",
        source="Swiss Financial Market Supervisory Authority (FINMA); UBS; Reuters",
        region="global", impact="high",
        tags=["banking_crisis", "credit_suisse", "ubs"],
    ),
]


# ── Lookup Functions ──────────────────────────────────────────────────────

def get_all_events() -> list[MarketEvent]:
    """Return the full curated event catalog."""
    return EVENTS


def get_events_by_category(category: str) -> list[MarketEvent]:
    """Return events matching a specific category."""
    cat_lower = category.lower().strip()
    return [e for e in EVENTS if e.category == cat_lower]


def get_events_by_tag(tag: str) -> list[MarketEvent]:
    """Return events containing a specific tag."""
    tag_lower = tag.lower().strip()
    return [e for e in EVENTS if tag_lower in [t.lower() for t in e.tags]]


def get_events_for_date(dt: date) -> list[MarketEvent]:
    """Return all events active on a specific date."""
    return [e for e in EVENTS if e.start_date <= dt <= (e.end_date or e.start_date)]


def get_events_in_range(start: date, end: date) -> list[MarketEvent]:
    """Return events overlapping with a date range."""
    return [
        e for e in EVENTS
        if e.start_date <= end and (e.end_date or e.start_date) >= start
    ]


def search_events(query: str) -> list[MarketEvent]:
    """Search events by name, description, or tags (case-insensitive)."""
    q = query.lower()
    results = []
    for e in EVENTS:
        if (q in e.name.lower()
                or q in e.description.lower()
                or q in e.category.lower()
                or any(q in t.lower() for t in e.tags)):
            results.append(e)
    return results


# ── Daily Event Tagging ──────────────────────────────────────────────────

def tag_dataframe(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Add event metadata columns to a daily price DataFrame.

    New columns added:
      - event_tags      : pipe-separated category list (e.g. "war|oil_shock")
      - event_names     : pipe-separated event names active that day
      - event_count     : number of concurrent events
      - event_impact    : highest impact level active (extreme > high > medium > low)
      - event_categories: set of distinct categories active that day (as pipe-separated string)
    """
    df = df.copy()

    impact_rank = {"low": 1, "medium": 2, "high": 3, "extreme": 4}

    # Pre-build a mapping: date → list[MarketEvent]
    # For efficiency, iterate events first and build sparse mapping
    date_events: dict[date, list[MarketEvent]] = {}
    for evt in EVENTS:
        start = evt.start_date
        end = evt.end_date or evt.start_date
        current = start
        while current <= end:
            date_events.setdefault(current, []).append(evt)
            current += timedelta(days=1)

    tags_list = []
    names_list = []
    counts_list = []
    impacts_list = []
    categories_list = []

    for _, row in df.iterrows():
        dt = pd.Timestamp(row[date_col]).date()
        active = date_events.get(dt, [])
        if active:
            cats = sorted(set(e.category for e in active))
            names = [e.name for e in active]
            max_impact = max((impact_rank.get(e.impact, 0) for e in active), default=0)
            impact_label = {v: k for k, v in impact_rank.items()}.get(max_impact, "none")

            tags_list.append("|".join(cats))
            names_list.append("|".join(names))
            counts_list.append(len(active))
            impacts_list.append(impact_label)
            categories_list.append("|".join(cats))
        else:
            tags_list.append("")
            names_list.append("")
            counts_list.append(0)
            impacts_list.append("none")
            categories_list.append("")

    df["event_tags"] = tags_list
    df["event_names"] = names_list
    df["event_count"] = counts_list
    df["event_impact"] = impacts_list
    df["event_categories"] = categories_list

    tagged_days = sum(1 for c in counts_list if c > 0)
    logger.info("Event tagging: %d/%d trading days have event metadata", tagged_days, len(df))

    return df


# ── Event-Period Market Analysis ──────────────────────────────────────────

def analyze_market_during_events(
    df: pd.DataFrame,
    category: str | None = None,
    tag: str | None = None,
    price_col: str = "Close",
    date_col: str = "Date",
) -> dict:
    """
    Analyze how the market (Nifty 50) performed during events of a given
    category or tag.

    Returns a dictionary with:
      - events analysed (list of event names)
      - stats for each event period (return, volatility, max drawdown)
      - aggregate stats across all event periods
      - comparison with non-event periods
    """
    # Filter events
    if category:
        events = get_events_by_category(category)
    elif tag:
        events = get_events_by_tag(tag)
    else:
        events = EVENTS

    if not events:
        return {"error": f"No events found for category='{category}', tag='{tag}'"}

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if "simple_return" not in df.columns:
        df["simple_return"] = df[price_col].pct_change()

    event_results = []
    all_event_dates = set()

    for evt in events:
        start = pd.Timestamp(evt.start_date)
        end = pd.Timestamp(evt.end_date or evt.start_date)

        # For single-day events, extend window to ±1 trading day for return calc
        is_single_day = (evt.end_date is None or evt.end_date == evt.start_date)
        if is_single_day:
            # Use 3-day window centred on event day
            lookup_start = start - pd.Timedelta(days=5)  # look back 5 calendar days to find trading days
            lookup_end = end + pd.Timedelta(days=5)
            mask = (df[date_col] >= lookup_start) & (df[date_col] <= lookup_end)
            nearby = df[mask]
            # Find the event day and surrounding trading days
            evt_idx = nearby[nearby[date_col] >= start].index
            if len(evt_idx) == 0:
                continue
            center = evt_idx[0]
            local_start = max(nearby.index[0], center - 1)
            local_end = min(nearby.index[-1], center + 1)
            period_df = df.loc[local_start:local_end]
        else:
            mask = (df[date_col] >= start) & (df[date_col] <= end)
            period_df = df[mask]

        if len(period_df) < 1:
            continue

        all_event_dates.update(period_df[date_col].dt.date.tolist())

        start_price = period_df[price_col].iloc[0]
        end_price = period_df[price_col].iloc[-1]
        period_return = (end_price / start_price - 1) * 100 if len(period_df) > 1 else 0
        daily_returns = period_df["simple_return"].dropna()
        vol = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0

        # Max drawdown
        cummax = period_df[price_col].cummax()
        drawdown = (period_df[price_col] - cummax) / cummax
        max_dd = drawdown.min() * 100

        event_results.append({
            "name": evt.name,
            "category": evt.category,
            "region": evt.region,
            "impact": evt.impact,
            "start": str(evt.start_date),
            "end": str(evt.end_date or evt.start_date),
            "trading_days": len(period_df),
            "return_pct": round(period_return, 2),
            "annualised_vol_pct": round(vol, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
        })

    if not event_results:
        return {"error": "No events had enough data for analysis"}

    # Aggregate stats across event periods
    returns = [r["return_pct"] for r in event_results]
    vols = [r["annualised_vol_pct"] for r in event_results]
    drawdowns = [r["max_drawdown_pct"] for r in event_results]

    avg_return = np.mean(returns)
    median_return = np.median(returns)
    pct_negative = sum(1 for r in returns if r < 0) / len(returns) * 100

    # Non-event periods comparison
    event_mask = df[date_col].dt.date.isin(all_event_dates)
    non_event_returns = df.loc[~event_mask, "simple_return"].dropna()
    event_daily_returns = df.loc[event_mask, "simple_return"].dropna()

    non_event_ann_return = non_event_returns.mean() * 252 * 100 if len(non_event_returns) > 0 else 0
    event_ann_return = event_daily_returns.mean() * 252 * 100 if len(event_daily_returns) > 0 else 0
    non_event_vol = non_event_returns.std() * np.sqrt(252) * 100 if len(non_event_returns) > 1 else 0
    event_vol = event_daily_returns.std() * np.sqrt(252) * 100 if len(event_daily_returns) > 1 else 0

    analysis = {
        "query": {
            "category": category,
            "tag": tag,
            "events_found": len(events),
            "events_with_data": len(event_results),
        },
        "event_periods": event_results,
        "aggregate": {
            "avg_period_return_pct": round(avg_return, 2),
            "median_period_return_pct": round(median_return, 2),
            "pct_periods_negative": round(pct_negative, 1),
            "avg_annualised_vol_pct": round(np.mean(vols), 2),
            "worst_drawdown_pct": round(min(drawdowns), 2),
            "best_return_pct": round(max(returns), 2),
            "worst_return_pct": round(min(returns), 2),
        },
        "comparison": {
            "during_events": {
                "annualised_return_pct": round(event_ann_return, 2),
                "annualised_vol_pct": round(event_vol, 2),
                "trading_days": int(event_mask.sum()),
            },
            "outside_events": {
                "annualised_return_pct": round(non_event_ann_return, 2),
                "annualised_vol_pct": round(non_event_vol, 2),
                "trading_days": int((~event_mask).sum()),
            },
        },
    }

    return analysis


def get_event_summary() -> dict:
    """Return a summary of all events in the catalog, grouped by category."""
    summary = {}
    for cat, desc in CATEGORIES.items():
        cat_events = get_events_by_category(cat)
        summary[cat] = {
            "description": desc,
            "count": len(cat_events),
            "events": [
                {
                    "name": e.name,
                    "dates": f"{e.start_date} to {e.end_date}" if e.end_date else str(e.start_date),
                    "impact": e.impact,
                    "region": e.region,
                }
                for e in cat_events
            ],
        }
    return summary
