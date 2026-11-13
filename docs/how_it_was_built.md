# How this was built: everything tried, kept and dropped

One map of the whole project, in plain words. **Green** = kept. **Grey** = tried and dropped, with the reason.
Each box names the notebook where it happens.

```mermaid
flowchart TB
    classDef keep fill:#e8f5e9,stroke:#2e7d32,color:#1b3a1f
    classDef drop fill:#f2f2f2,stroke:#9e9e9e,color:#555,stroke-dasharray: 4 3
    classDef data fill:#fde7f3,stroke:#e2178f,color:#4a0d2e
    classDef out fill:#011c45,stroke:#011c45,color:#ffffff

    subgraph D["1 · Data"]
        D1["RR's data: 883 matches, IPL + T20Is 2022–26"]:::data
        D2["Fix extras, remove super overs, use player IDs<br/>→ 867 matches, 200,961 balls (nb 1)"]:::keep
        D3["Rebuild the situation before every ball<br/>score, wickets, balls left, target, ground, season (nb 2)"]:::keep
        D4["Outside data: 4,005 Cricsheet matches<br/>12 duplicates removed → 3,993, used ONLY to test (nb 3)"]:::data
        D1 --> D2 --> D3
    end

    subgraph W["2 · Who wins? (chance of winning before every ball, nb 4–5)"]
        W1["Decision tree, random forest, extra trees,<br/>XGBoost, LightGBM, CatBoost, KNN<br/>✗ memorised matches"]:::drop
        W2["Two stacked ensembles, one average<br/>✗ no better, overfitted more"]:::drop
        W3["Logistic with curves on every input<br/>✗ too sure of itself"]:::drop
        W4["WASP-style backward calculation<br/>✗ too pessimistic at the end of chases"]:::drop
        W5["Logistic regression, curves on the game state only<br/>✓ best on new matches, best calibrated"]:::keep
    end

    subgraph P["3 · What is pressure? (nb 6)"]
        P1["38 signs of pressure<br/>rate above 10, dot streaks, new batter, collapse, knockout…"]:::keep
        P2["3 tests each: changes performance?<br/>decides matches? true every season?"]:::keep
        P3["Clustering, PCA, simple rules,<br/>2 published indexes<br/>✗ scored lower"]:::drop
        P4["Impact × how big the match is<br/>✓ top 20% of balls = pressure balls"]:::keep
        P5["Separate cut-off for each innings<br/>✗ first-innings extra balls barely mattered"]:::drop
        P1 --> P2 --> P4
        P2 --> P3
    end

    subgraph A["4 · The average player (nb 4)"]
        A1["Lookup table · logistic regression<br/>✗ less accurate on new matches"]:::drop
        A2["LightGBM, 41 situation inputs, never sees names<br/>✓ best on new matches, rates within 0.04%"]:::keep
        A3["LightGBM with team strength (64 inputs)<br/>✗ tiny gain, describes teams not situations"]:::drop
        A4["Bowlers compared with all bowlers<br/>✗ put 9 spinners in the top 10"]:::drop
        A5["Bowlers compared with their own type<br/>✓ pace v pace, spin v spin"]:::keep
    end

    subgraph R["5 · Rating players (nb 7–11)"]
        R1["Pressure minus his own usual level<br/>✗ does not repeat: looks like luck"]:::drop
        R2["4 ways to value a ball<br/>✓ runs for bowlers, skill profile for batters"]:::keep
        R3["Ridge plus-minus, pressure balls only,<br/>a blend, a consistency bonus<br/>✗ lower on hidden matches"]:::drop
        R4["Empirical Bayes, every ball weighted by pressure<br/>✓ small samples pulled to average, 60-ball minimum"]:::keep
        R5["Roles from performance clusters<br/>✗ mislabelled finishers"]:::drop
        R6["Roles from how players are used<br/>✓ labels only, never used to rank"]:::keep
    end

    subgraph T["6 · Choosing and testing (nb 11–15)"]
        T1["52 hidden-match tests<br/>✓ every setting chosen here"]:::keep
        T2["1,000 rebuilds on reshuffled matches<br/>✓ 'stays top 10' for every player"]:::keep
        T3["2026 alone: bowlers ✓ held up · batters ✗ did not<br/>(one season is mostly luck for batters)"]:::keep
        T4["3,993 outside matches<br/>✓ both top 10s beat average · batters weaker"]:::keep
        T5["Only players active in 2025–26"]:::keep
    end

    OUT(["Top 10 batters · Top 10 bowlers"]):::out

    D3 --> W5
    D3 --> A2
    W5 --> P4
    P4 --> R4
    A2 --> R2 --> R4
    A5 --> R2
    R4 --> T1 --> T2 --> T5 --> OUT
    D4 --> T4
    OUT -.-> T3
    OUT -.-> T4
```

## The same story in seven lines

1. **Clean the data.** Fix extras, drop super overs, use player IDs: 867 matches, 200,961 balls.
2. **Predict who wins before every ball.** 11 models were tried. The simple logistic regression won, because the others memorised matches.
3. **Define pressure.** 38 signs, 3 tests, 8 definitions. Impact × how big the match is won. The top 20% of balls are pressure balls.
4. **Learn the average player.** LightGBM, never seeing names. Bowlers are only compared with their own type.
5. **Rate players.** Every ball counts, pressure balls count most, and small samples are pulled to average. "Rising above your own level" was tested and dropped, because it looked like luck.
6. **Choose and test.** 52 hidden-match tests, 1,000 rebuilds, the 2026 season and 3,993 outside matches.
7. **Answer.** Top 10 batters and bowlers, active players only.
