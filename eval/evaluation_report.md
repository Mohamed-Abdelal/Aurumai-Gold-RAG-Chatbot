# AurumAI Evaluation Report: With RAG vs. Without RAG

- **Evaluated at (UTC):** `2026-11-20 12:08:40 UTC`
- **Groq LLM Model:** `openai/gpt-oss-120b`
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Retrieval Settings:** `Top-K = 4`, `min_similarity = 0.25`
- **Total Questions:** `10` (8 in-scope + 2 out-of-scope)

## Summary Metrics

| Metric | With RAG (AurumAI) | Without RAG (Baseline) | Improvement |
| :--- | :---: | :---: | :---: |
| **In-Scope Retrieval Hit@4** | **100.0%** | N/A | — |
| **Mean Top-1 Cosine Similarity (In-Scope)** | `0.7256` | N/A | — |
| **Mean Key-Fact / Refusal Recall** | **98.0%** | 30.9% | +67.1 pp |
| **Out-of-Scope Safe Refusal Rate** | **100.0%** | 0.0% | +100.0 pp |
| **Mean Correctness Score (1–5)** | **5.00 / 5** | 1.90 / 5 | +3.10 |
| **Mean Relevance Score (1–5)** | **5.00 / 5** | 4.00 / 5 | +1.00 |
| **Hallucination / Boundary-Violation Rate** | **0.0%** | 20.0% | -20.0 pp |
| **Mean Response Latency (ms)** | 7843 ms | 2298 ms | — |

## Per-Question Comparison Table

| # | Type | Target File(s) | Hit@K / Refusal | Top-1 Sim | Recall (RAG vs No-RAG) | Correctness (RAG vs No-RAG) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| Q1 | `fact` | `02_gold_etfs_and_spot_trackers.txt` | Yes | 0.658 | 100% vs 40% | 5/5 vs 2/5 |
| Q2 | `fact` | `07_60_20_20_balanced_portfolio_with_gold.txt` | Yes | 0.677 | 100% vs 80% | 5/5 vs 4/5 |
| Q3 | `fact` | `14_2026_gold_market_rally_and_pullback_dynamics.txt` | Yes | 0.707 | 100% vs 60% | 5/5 vs 2/5 |
| Q4 | `fact` | `16_egypt_24k_gold_bars_vs_21k_gold_pounds.txt` | Yes | 0.829 | 100% vs 60% | 5/5 vs 1/5 |
| Q5 | `fact` | `17_egypt_making_charges_masnaeya_and_cashback.txt` | Yes | 0.706 | 100% vs 0% | 5/5 vs 2/5 |
| Q6 | `fact` | `18_egypt_gold_mutual_funds_az_gold_and_regulated_options.txt` | Yes | 0.797 | 100% vs 29% | 5/5 vs 1/5 |
| Q7 | `multi_doc_profile` | `20_gold_investor_decision_matrix.txt`, `18_egypt_gold_mutual_funds_az_gold_and_regulated_options.txt`, `17_egypt_making_charges_masnaeya_and_cashback.txt` | Yes | 0.750 | 100% vs 20% | 5/5 vs 2/5 |
| Q8 | `multi_doc_profile` | `20_gold_investor_decision_matrix.txt`, `09_rebalancing_and_investment_horizon_for_gold.txt`, `12_bid_ask_spreads_and_hidden_transaction_costs.txt` | Yes | 0.680 | 80% vs 20% | 5/5 vs 2/5 |
| Q9 | `out_of_scope` | _Out of Scope_ | Refused | 0.349 | 100% vs 0% | 5/5 vs 2/5 |
| Q10 | `out_of_scope` | _Out of Scope_ | Refused | 0.518 | 100% vs 0% | 5/5 vs 1/5 |

## Detailed Qualitative Comparisons

### Q1 [fact]: What are the annual expense ratios of GLD, IAU, and GLDM, and what typical bid-ask spread do major Gold ETFs offer?
- **Target File(s):** `02_gold_etfs_and_spot_trackers.txt`
- **Verified Reference:** SPDR Gold Shares (GLD) charges 0.40% per year, iShares Gold Trust (IAU) charges 0.25% per year, and SPDR Gold MiniShares (GLDM) charges 0.10% per year. Major Gold ETFs typically offer razor-thin bid-ask spreads of 0.02% to 0.08% on stock exchanges.
- **Top-1 Cosine Sim:** `0.6582`

**With RAG (AurumAI):**
> - **Annual expense ratios** (as listed for the three major gold ETFs):  
>   - **GLD (SPDR Gold Shares):** 0.40% per year  
>   - **IAU (iShares Gold Trust):** 0.25% per year  
>   - **GLDM (SPDR Gold MiniShares):** 0.10% per year  
> 
> - **Typical bid‑ask spread for major gold ETFs:** very narrow, generally **0.02 % – 0.08 %** on the major exchanges.  
> 
> **Sources**  
> - Expense ratios and spread range are detailed in the “Gold ETFs and Spot Price Trackers Explained” article [**02_gold_etfs_and_spot_trackers.txt**].  
> 
> *Disclaimer: This information is for educational purposes only and does not constitute licensed financial advice.*

**Without RAG (Baseline):**
> **Expense ratios (annual)**  
> 
> | ETF | Expense Ratio |
> |-----|---------------|
> | **GLD** (SPDR Gold Shares) | **0.40 %** |
> | **IAU** (iShares Gold Trust) | **0.25 %** |
> | **GLDM** (SPDR Gold Mini Shares) | **0.18 %** |
> 
> *(All figures are the fund‑level management fees disclosed in each ETF’s prospectus and are expressed as a percentage of assets under management per year.)*  
> 
> **Typical bid‑ask spreads**
> 
> - For the three most‑liquid gold ETFs (GLD, IAU, GLDM) the **average bid‑ask spread** in normal market conditions is roughly **5–10 basis points** of the fund’s net asset value (NAV), which translates to about **$0.05–$0.10 per share** for GLD and IAU (which trade around $180–$200) and a similar proportion for GLDM (which trades around $30–$40).  
> - During periods of heightened volatility or low liquidity the spread can widen to **15–20 bps** (≈ $0.15–$0.20 per share), but for most trading days the 5‑10 bps range is typical for “major” gold ETFs.  
> 
> So, in summary: GLD = 0.40 %, IAU = 0.25 %, GLDM = 0.18 %; and you can generally expect a bid‑ask spread of about **5–10 cents per share (≈ 5–10 bps)** for these funds.

- *Evaluator Note (RAG):* Provides the exact expense ratios and bid‑ask spread range as in the reference. (Correctness: 5/5, Relevance: 5/5)
- *Evaluator Note (No-RAG):* Expense ratio for GLDM and bid‑ask spread range are inaccurate compared to the reference. (Correctness: 2/5, Relevance: 5/5)

### Q2 [fact]: How is the 60/20/20 balanced portfolio split across asset classes, and what structural flaw in the traditional 60/40 model did 2022-2026 inflation expose?
- **Target File(s):** `07_60_20_20_balanced_portfolio_with_gold.txt`
- **Verified Reference:** The 60/20/20 portfolio allocates 60% to equities, 20% to fixed income and cash equivalents, and 20% to gold and hard-asset alternatives. Inflationary episodes from 2022 through 2026 exposed that stocks and long-duration bonds can decline at the same time when inflation and interest rates remain elevated, breaking 60/40's negative correlation.
- **Top-1 Cosine Sim:** `0.6772`

**With RAG (AurumAI):**
> The **60/20/20 balanced portfolio** allocates assets as follows:
> 
> | Asset class | Allocation |
> |-------------|------------|
> | **Equities** (global and domestic stock index funds or quality dividend equities) | **60 %** |
> | **Fixed‑income & cash equivalents** (short‑term Treasury bills, high‑yield money‑market funds, inflation‑linked bonds) | **20 %** |
> | **Gold & hard‑asset alternatives** (physical bullion, gold ETFs, regulated gold funds, optionally combined with other commodities or real‑asset exposure) | **20 %** |
> 
> This structure is described in the AurumAI Portfolio Research article [07_60_20_20_balanced_portfolio_with_gold.txt]【Source 1】.
> 
> **Structural flaw in the traditional 60/40 model (60 % equities / 40 % bonds):**  
> During the inflationary episodes of 2022‑2026, central banks kept interest rates high or raised them. That environment caused **both stocks and long‑duration bonds to fall at the same time**, breaking the negative correlation that the 60/40 split relied on for diversification. Consequently, the portfolio lost its built‑in hedge against inflation‑driven market stress【Source 1】.
> 
> *Disclaimer: This information is for educational purposes only and does not constitute licensed financial advice.*

**Without RAG (Baseline):**
> **60/20/20 “Balanced” Portfolio – Asset‑Class Breakdown**
> 
> | Allocation | Typical Asset Class(s) | What It Provides |
> |------------|------------------------|------------------|
> | **60 %**   | **Equities** (global stocks) – a mix of U.S. large‑cap, international developed, and emerging‑market equities. | Long‑term growth, capital appreciation, and inflation‑participation. |
> | **20 %**   | **Real Assets** – broadly split between **real‑estate (REITs or direct property)** and **commodities (including energy, metals, agriculture)**. | Direct exposure to price‑inflation drivers, diversification, and a hedge against rising consumer prices. |
> | **20 %**   | **Fixed Income** – a blend of **inflation‑protected bonds (e.g., TIPS, global inflation‑linked sovereigns)** and **short‑duration nominal bonds** (investment‑grade corporates, high‑quality sovereigns). | Income, capital preservation, and a low‑correlation buffer to equity volatility while still offering some inflation protection. |
> 
> *The exact split between REITs vs. commodities, or TIPS vs. nominal bonds, can be tuned to the investor’s risk tolerance, tax situation, and regional preferences, but the core idea is 60 % growth (stocks), 20 % inflation‑hedge (real assets), 20 % income/deflation‑buffer (inflation‑aware bonds).*
> 
> ---
> 
> ### Structural Flaw Exposed in the Traditional 60/40 Model (2022‑2026)
> 
> **Flaw:** *The 60 % equity / 40 % nominal‑bond mix assumes that bonds will act as a stable, low‑correlation “defensive” anchor that preserves capital and provides income when equities falter. It relies on a relatively stable, low‑inflation environment where nominal yields remain positive and real returns on bonds stay modestly positive.*
> 
> **What 2022‑2026 Inflation Revealed:**
> 
> | Issue | How It Manifested |
> |-------|-------------------|
> | **Rising Real Yields & Negative Real Returns on Nominal Bonds** | Inflation surged to 7‑9 % in many economies, while central banks kept policy rates high. Nominal bond prices fell sharply, delivering large capital losses and, after inflation adjustment, negative real returns. |
> | **Higher Correlation Between Equities and Bonds** | As investors fled both asset classes during rate‑hike cycles, equities and nominal bonds moved down together, eroding the diversification benefit that the 60/40 model counts on. |
> | **Liquidity & Duration Mismatch** | Long‑duration bonds suffered steep price drops, while equity markets remained volatile. The “safe‑haven” bond leg could not be liquidated without locking in losses, undermining the portfolio’s ability to rebalance. |
> | **Insufficient Inflation Hedge** | The 40 % bond allocation was largely nominal, offering little protection against the rapid rise in consumer prices, so the overall portfolio’s real return was dragged down. |
> 
> **Bottom line:** The 60/40 model’s implicit belief that “bonds = safety” broke down when inflation turned aggressive and interest rates rose sharply. The model lacked a built‑in, real‑asset or inflation‑linked component to preserve purchasing power, leading investors to experience simultaneous equity‑bond drawdowns and negative real returns. The 60/20/20 framework addresses this by allocating a dedicated 20 % to real assets and using inflation‑protected bonds for the fixed‑income slice.

- *Evaluator Note (RAG):* Accurately repeats the split and flaw described in the reference. (Correctness: 5/5, Relevance: 5/5)
- *Evaluator Note (No-RAG):* Matches the split and flaw closely, though asset class wording differs slightly. (Correctness: 4/5, Relevance: 5/5)

### Q3 [fact]: How large have sharp 2026 gold pullbacks been from record highs, how many metric tons per year have central banks bought, and what catalysts trigger those pullbacks?
- **Target File(s):** `14_2026_gold_market_rally_and_pullback_dynamics.txt`
- **Verified Reference:** Even inside a structural bull trend backed by central banks buying 800 to 1,000 metric tons annually, 2026 has witnessed sharp multi-week pullbacks of 8% to 15% from record highs. The four triggers are real-yield repricing, crowded speculative positioning, stronger U.S. dollar (DXY) bounces, and local FX premium compression.
- **Top-1 Cosine Sim:** `0.7070`

**With RAG (AurumAI):**
> - **Size of sharp 2026 pullbacks:**  
>   Pullbacks have typically ranged **from 8 % to 15 %** off local record highs, often occurring over a few weeks. [14_2026_gold_market_rally_and_pullback_dynamics.txt]
> 
> - **Central‑bank gold purchases:**  
>   Emerging‑market central banks have been buying **about 800 to 1,000 metric tons of gold each year**, providing a strong structural demand floor. [14_2026_gold_market_rally_and_pullback_dynamics.txt]
> 
> - **Catalysts that trigger those pullbacks:**  
>   1. **Real‑yield repricing** – stronger‑than‑expected inflation or employment data pushes bond yields up, prompting algorithmic futures funds to dump gold. [14_2026_gold_market_rally_and_pullback_dynamics.txt]  
>   2. **Crowded speculative positioning** – when futures net‑long positions and retail FOMO surge, modest profit‑taking can cascade into stop‑loss liquidations and margin calls. [14_2026_gold_market_rally_and_pullback_dynamics.txt]  
>   3. **Stronger U.S. dollar bounces** – short‑term rallies in the DXY exert downward pressure on USD‑denominated spot gold. [14_2026_gold_market_rally_and_pullback_dynamics.txt]  
>   4. **Local FX premium compression (e.g., Egypt)** – when domestic gold trades above global USD parity due to currency anxiety, inflows of FX liquidity or improved currency stability can quickly compress the local premium, amplifying pullbacks in local‑currency terms. [14_2026_gold_market_rally_and_pullback_dynamics.txt]
> 
> *Disclaimer: This information is for educational purposes only and does not constitute licensed financial advice.*

**Without RAG (Baseline):**
> **Sharp 2026 gold pull‑backs from record highs**
> 
> | Year (2026) | Record‑high price (USD/oz) | Low after pull‑back (USD/oz) | Approx. % decline | Duration of pull‑back* |
> |-------------|---------------------------|-----------------------------|-------------------|------------------------|
> | **Q1‑Q2**   | $2,450 – $2,500 (all‑time high reached in early Jan) | $2,080 – $2,120 | **≈ 15 %** | 4‑5 weeks (mid‑Jan → early Feb) |
> | **Q3**      | $2,300 – $2,350 (high after mid‑summer rally) | $1,950 – $2,000 | **≈ 14 %** | 3‑4 weeks (late Aug → mid‑Sept) |
> | **Q4**      | $2,200 – $2,250 (late‑Nov peak) | $1,880 – 1,920 | **≈ 14‑15 %** | 3‑4 weeks (early Dec → late Dec) |
> 
> \*Duration is measured from the peak to the trough of the sharpest intra‑quarter correction. The declines are “sharp” in the sense that they exceed the typical 5‑8 % intra‑month volatility seen in gold and are accompanied by a noticeable rise in short‑term futures‑based short‑interest.
> 
> ---
> 
> ### Central‑bank buying (net purchases) – metric tons per year
> 
> | Year | Net purchases (metric tons) | Sources / notes |
> |------|----------------------------|-----------------|
> | **2022** | ~ 380 t | IMF/World Gold Council (WGC) “Gold Demand Trends” |
> | **2023** | ~ 340 t | WGC, slight slowdown as many banks finished 2021‑22 buying cycles |
> | **2024** | ~ 310 t | WGC, continued modest buying, focus on diversification |
> | **2025** (estimate) | **≈ 300 t** | Trend line extrapolation; most major central banks still net buyers |
> | **2026** (estimate) | **≈ 295 t** | Same trend; a few banks (e.g., China, Russia) expected to pause or modestly sell, offset by others (Turkey, India) buying |
> 
> *The numbers are **net** purchases (purchases minus sales) reported by the World Gold Council. They are rounded to the nearest 10 t for simplicity.*
> 
> ---
> 
> ### Typical catalysts that trigger the sharp pull‑backs
> 
> | Catalyst | How it affects gold | Why it produces a ~15 % drop |
> |----------|--------------------|------------------------------|
> | **Rapid US‑rate hikes / hawkish Fed commentary** | Higher real yields make non‑yielding gold less attractive. | A 25‑50 bp surprise in the Fed Funds rate (or a forward‑looking “higher‑for‑longer” stance) can lift 10‑year Treasury yields 30‑50 bps, instantly cutting gold’s carry and prompting short‑term profit‑taking. |
> | **Strong US dollar rally** | Gold is priced in dollars; a stronger dollar raises the effective price for holders of other currencies. | A 3‑4 % USD index gain over a few weeks (often driven by safe‑haven flight to the dollar) can shave 10‑12 % off gold’s price. |
> | **Resolution of a major geopolitical tension** | Gold often spikes on uncertainty; a quick de‑escalation removes the “fear premium.” | Example: a cease‑fire in a high‑risk region (e.g., Middle East) or a diplomatic breakthrough on a trade war can cause a rapid risk‑off shift. |
> | **Large‑scale profit‑taking by futures traders** | Futures markets amplify price moves; a surge in short‑interest or a “short‑covering rally” reversal can trigger a cascade of stop‑loss orders. | When short‑interest hits > 30 % of open interest (as seen in Q1‑2026), a modest price rise can force a rapid unwind, pushing spot gold down sharply. |
> | **Unexpected macro data (inflation, employment)** | Better‑than‑expected data reduces inflation fears and supports rate‑hike expectations. | A CPI print 0.5 % lower than consensus or a strong US jobs report can lift yields and the dollar, prompting a 10‑15 % pull‑back. |
> | **Central‑bank policy shift** | If a major holder announces a pause or modest sale, market sentiment can swing. | In Q3‑2026

- *Evaluator Note (RAG):* Accurately mirrors the reference answer with all key details. (Correctness: 5/5, Relevance: 5/5)
- *Evaluator Note (No-RAG):* Provides detailed but inaccurate pullback, central‑bank, and trigger data not matching the reference. (Correctness: 2/5, Relevance: 4/5)

### Q4 [fact]: In Egypt, how many grams does a standard 21K gold pound weigh, how much pure 24K gold does it contain, and why should buyers prefer branded sealed pounds over unbranded baladi pounds?
- **Target File(s):** `16_egypt_24k_gold_bars_vs_21k_gold_pounds.txt`
- **Verified Reference:** A standard Egyptian 21K gold pound weighs 8 grams of 21-karat gold (87.5% purity), containing exactly 7.0 grams of pure 24K gold plus 1 gram of alloy. Buyers should prefer branded, factory-sealed, officially hallmarked pounds over unbranded baladi pounds because they guarantee exact purity, include tamper-evident assay cards, and qualify for partial making-charge cashback upon resale.
- **Top-1 Cosine Sim:** `0.8290`

**With RAG (AurumAI):**
> - **Standard weight of a 21‑karat Egyptian gold pound:** 8 grams of 21K gold (87.5 % purity) 【16_egypt_24k_gold_bars_vs_21k_gold_pounds.txt】  
> - **Pure 24‑karat gold contained:** Exactly 7.0 grams of pure 24K gold (the remaining 1 gram is copper/alloy for durability) 【16_egypt_24k_gold_bars_vs_21k_gold_pounds.txt】  
> 
> **Why choose branded, factory‑sealed gold pounds rather than unbranded “baladi” pounds**
> 
> 1. **Verified purity:** Branded pounds are hallmarked by the Egyptian Assay and Weights Administration and carry a clear karat stamp (875 for 21K) plus the accredited manufacturer’s logo, guaranteeing the exact 21K purity 【19_egypt_gold_invoice_hallmark_and_buyback_checklist.txt】.  
> 2. **Tamper‑evident packaging & assay card:** They come in sealed, tamper‑evident packs with an assay card, making it easy to prove authenticity at resale.  
> 3. **Resale advantages:** Branded pounds qualify for partial making‑charge cashback when sold back and enjoy **universal instant liquidity** at virtually every jewelry shop (“sagha”) across Egypt, because the daily retail gold price is quoted in 21K per gram 【16_egypt_24k_gold_bars_vs_21k_gold_pounds.txt】.  
> 4. **Lower resale discount:** Unbranded baladi pounds are harder to verify and often face steeper resale discounts, reducing the net return for investors.  
> 
> *Disclaimer: This information is for educational purposes only and does not constitute licensed financial advice.*

**Without RAG (Baseline):**
> **Standard 21‑karat (21K) gold “pound” in Egypt**
> 
> | Item | Typical value |
> |------|---------------|
> | **Weight of the whole pound** | **≈ 12.5 g** (the legal “pound” for jewelry is 12.5 g, not the 453.6 g avoirdupois pound) |
> | **Pure 24‑karat gold content** | **≈ 11.9 g** of 24K gold |
> 
> **How the numbers are derived**
> 
> 1. **21K gold = 87.5 % pure gold** (21 parts gold / 24 total parts).  
> 2. **12.5 g × 0.875 = 10.9375 g** – this is the amount of pure gold in a *perfect* 21K alloy.  
> 3. In practice, most Egyptian “pounds” are slightly heavier (≈ 12.8 g) because manufacturers add a small excess to compensate for handling loss and to meet consumer expectations.  
> 4. Using the common market weight of **12.8 g**:  
>    - 12.8 g × 0.875 = **11.2 g** of 24K gold.  
>    - Many dealers round this to **≈ 11.9 g** when quoting the “pure gold” value, because they assume the pound is 13.6 g (the weight of a *baladi* 21K pound) and then apply the 87.5 % factor.  
> 
> The most reliable figure for a **standard, legally‑defined 21K gold pound** is **12.5 g total weight, containing about 10.9 g of pure gold**. Market‑price listings often use the slightly higher “effective” pure‑gold weight (≈ 11.9 g) to reflect the typical commercial weight you’ll actually receive.
> 
> ---
> 
> ### Why buyers should prefer **branded, sealed pounds** over **unbranded “baladi” pounds**
> 
> | Aspect | Branded, sealed pound | Unbranded “baladi” pound |
> |--------|----------------------|--------------------------|
> | **Guarantee of weight** | Certified by the manufacturer; sealed packaging shows the exact weight (usually 12.5 g ± 0.02 g). | Weight is measured by the seller on the spot; can vary widely (often 12 g–13.6 g) and is not independently verified. |
> | **Purity verification** | Hallmarked with the karat stamp (21K) and often a serial/lot number; the seal is tamper‑evident. | No official hallmark; purity is claimed verbally and may be tested only after purchase. |
> | **Traceability** | You can contact the brand or check the serial number if there’s a dispute. | No traceability; if the gold is under‑weight or of lower purity, you have little recourse. |
> | **Consumer confidence** | Brands (e.g., Al‑Ahram, Al‑Mansour, Gold‑Standard) have reputations and are subject to periodic audits by the Egyptian Ministry of Trade. | “Baladi” pounds are sold by informal dealers; quality control is informal and inconsistent. |
> | **Resale value** | Higher resale price because the market trusts the brand’s certification. | Often discounted on resale because buyers demand a “purity test” or a price reduction for the risk. |
> | **Legal protection** | Covered by consumer‑protection laws; you can file a complaint against the brand if the seal is broken or the weight is off. | Limited legal recourse; disputes are usually settled informally. |
> | **Risk of fraud** | Very low; tampering with a sealed package is obvious. | Higher risk of “short‑weight” fraud, mixing of lower‑karat gold, or even counterfeit metal. |
> 
> **Bottom line:**  
> A **branded, sealed 21K gold pound** gives you a **documented, verifiable weight and purity**, reduces the chance of short‑weight or impurity fraud, and typically commands a better resale price. Unbranded “baladi” pounds may be cheaper upfront, but the hidden risks (weight variance, unknown purity, no traceability) often make them a poorer value in the long run.

- *Evaluator Note (RAG):* Provides the exact weight, pure gold content, and reasons to prefer branded sealed pounds as in the reference. (Correctness: 5/5, Relevance: 5/5)
- *Evaluator Note (No-RAG):* Provides incorrect weight and gold content compared to the reference, though stays on topic. (Correctness: 1/5, Relevance: 4/5)

### Q5 [fact]: How do Egypt making charges (masna'eya) compare between a 1g 24K bar and a 10g-20g 24K bar, and what happens to the cashback refund if you open the plastic assay packaging?
- **Target File(s):** `17_egypt_making_charges_masnaeya_and_cashback.txt`
- **Verified Reference:** In Egypt, a 1g 24K bar typically carries EGP 150 to EGP 185+ per gram in masna'eya and stamp fees, whereas 10g to 20g 24K bars typically cost EGP 55 to EGP 70 per gram. Accredited brands offer a 35% to 50% masna'eya cashback upon resale ONLY if the plastic assay card stays completely sealed; opening it turns the bar into kasr (opened gold) and forfeits the cashback refund.
- **Top-1 Cosine Sim:** `0.7063`

**With RAG (AurumAI):**
> **Making‑charge (masnaʿeya) comparison**
> 
> | Bar size | Typical masnaʿeya per gram (EGP) | Typical total masnaʿeya* |
> |----------|--------------------------------|--------------------------|
> | **1 g 24K bar** | **EGP 150 – 185+** per gram (the highest per‑gram cost) | ≈ EGP 150 – 185 (all of the price is masnaʿeya‑plus‑stamp) |
> | **10 g – 20 g 24K bar** | **EGP 55 – 70** per gram (the “sweet‑spot” for retail savers) | ≈ EGP 550 – 1,400 total, far lower per‑gram than a 1 g bar |
> 
> *The total masnaʿeya is the per‑gram amount multiplied by the bar’s weight. The figures above are taken from the range reported for each size class in the Egyptian market — 1 g bars are the most expensive per gram, while 10 g–20 g bars are the most efficient 【17_egypt_making_charges_masnaeya_and_cashback.txt】.
> 
> **Cashback refund rule**
> 
> - Accredited Egyptian bullion manufacturers (e.g., BTC, Swiss Gold, Selema Gold) attach a **sealed‑plastic assay card** to each bar or gold pound.  
> - If the seal remains **100 % intact**, the dealer refunds **35 %–50 % of the paid masnaʿeya** (e.g., EGP 20 – 30+ per gram) when you resell the item back to an authorized distributor 【17_egypt_making_charges_masnaeya_and_cashback.txt】.  
> - **If the plastic packaging is opened or damaged**, the bar is considered “opened/broken gold” (*kasr*). The dealer **forfeits the cashback refund** and deducts the full making charge from any buy‑back price 【17_egypt_making_charges_masnaeya_and_cashback.txt】.
> 
> **Key takeaway for Egyptian investors**  
> Buy larger bars (10 g–20 g) to minimise the per‑gram masnaʿeya, and always keep the assay‑card seal untouched to preserve the valuable cashback refund.
> 
> *Disclaimer: This information is for educational purposes only and does not constitute licensed financial advice.*

**Without RAG (Baseline):**
> **Masna‘eya (making‑charge) in Egypt**
> 
> | Gold bar size | Typical masna‘eya (making‑charge) |
> |---------------|-----------------------------------|
> | 1 g 24 K bar  | **≈ 0.5 % – 0.8 %** of the gold value (about 0.05 – 0.08 g of gold) |
> | 10 g – 20 g 24 K bar | **≈ 0.3 % – 0.5 %** of the gold value (about 0.03 – 0.10 g of gold) |
> 
> *Why the difference?*  
> The charge is expressed as a percentage of the bar’s weight/value, so larger bars benefit from economies of scale. A 1 g bar carries a slightly higher per‑gram charge than a 10 g or 20 g bar.
> 
> ---
> 
> **Cash‑back (refund) and the plastic assay packaging**
> 
> - The **cash‑back/refund** you receive when you return a gold bar (or sell it back to the dealer) is calculated on the **actual gold content** of the bar, not on the packaging.
> - **If you open the sealed plastic assay (the tamper‑proof wrapper):**
>   1. The bar is still considered “intact” as long as the assay label and serial number remain readable.
>   2. Most dealers will **still give you the full cash‑back** based on the bar’s weight and purity, because the assay sticker is the official proof of authenticity.
>   3. However, some dealers (especially those that guarantee a “sealed‑packaging” premium) may **deduct a small amount** (typically 0.1 % – 0.2 % of the bar’s value) for the loss of the original packaging, or they may refuse the premium altogether.
>   4. If the plastic is torn to the point that the assay label is damaged or the serial number becomes unreadable, the bar may be treated as “un‑assayed,” and the dealer could **refuse the cash‑back** or require a re‑assay, which can cost you extra.
> 
> **Bottom line**
> 
> - Larger bars have a lower per‑gram masna‘eya than 1 g bars.  
> - Opening the plastic assay packaging usually does **not** affect the cash‑back amount, provided the assay label stays intact; only a minor premium loss may be applied by some dealers. If the label is damaged, you risk losing the premium or having to re‑assay the bar.

- *Evaluator Note (RAG):* Provides the exact masna'eya ranges and cashback conditions matching the reference. (Correctness: 5/5, Relevance: 5/5)
- *Evaluator Note (No-RAG):* Provides unrelated percentage figures instead of the correct EGP per‑gram values and misstates the cashback rules. (Correctness: 2/5, Relevance: 4/5)

### Q6 [fact]: Who regulates and manages AZ-Gold in Egypt, what is its early-redemption fee schedule across Years 1, 2, 3, and after 3 years, and what minimum balance allows physical bullion redemption?
- **Target File(s):** `18_egypt_gold_mutual_funds_az_gold_and_regulated_options.txt`
- **Verified Reference:** AZ-Gold is regulated by Egypt's Financial Regulatory Authority (FRA) and managed by Azimut Egypt Asset Management. Its declining redemption fee is ~4.9% in Year 1, ~3.3% in Year 2, ~1.7% in Year 3, and 0% after 3 full years. Investors who accumulate at least 50 grams of gold equivalent can elect physical bullion delivery.
- **Top-1 Cosine Sim:** `0.7972`

**With RAG (AurumAI):**
> **Regulator & Manager**  
> - AZ‑Gold is a **Financial Regulatory Authority (FRA / الهيئة العامة للرقابة المالية)**‑regulated open‑ended precious‑metals fund.  
> - It is **managed by Azimut Egypt Asset Management** (the fund sponsor)【18_egypt_gold_mutual_funds_az_gold_and_regulated_options.txt】.  
> 
> **Early‑redemption fee schedule**  
> | Holding period | Redemption fee* |
> |----------------|-----------------|
> | Year 1 | ~4.9% |
> | Year 2 | ~3.3% |
> | Year 3 | ~1.7% |
> | After 3 full years | 0% (no fee) |
> 
> *Fees are applied to the redemption amount and decline the longer the investor stays in the fund【18_egypt_gold_mutual_funds_az_gold_and_regulated_options.txt】.  
> 
> **Physical bullion redemption threshold**  
> - Investors who accumulate **at least 50 grams of gold‑equivalent** in their AZ‑Gold balance can elect to receive physical bullion on redemption (subject to delivery procedures)【18_egypt_gold_mutual_funds_az_gold_and_regulated_options.txt】.  
> 
> *Disclaimer: This information is for educational purposes only and does not constitute licensed financial advice.*

**Without RAG (Baseline):**
> **Regulator & Manager**  
> - **Regulator:** The **Central Bank of Egypt (CBE)** is the authority that oversees and regulates all retail gold‑savings products in the country, including AZ‑Gold.  
> - **Manager/Issuer:** AZ‑Gold is issued and administered by **Al Ahly Capital Holding (through its subsidiary Al Ahly Capital – Gold Services)**, which operates the product on behalf of the Central Bank’s framework.
> 
> **Early‑Redemption (Early‑Exit) Fee Schedule**  
> | Holding Period | Early‑Redemption Fee* |
> |----------------|-----------------------|
> | **Year 1**      | 2.00 % of the redemption amount |
> | **Year 2**      | 1.50 % of the redemption amount |
> | **Year 3**      | 1.00 % of the redemption amount |
> | **After Year 3**| 0.50 % of the redemption amount |
> 
> \*The fee is applied to the **principal value** (the amount of gold you have purchased) at the time of redemption. The fee is deducted before the market price of gold is applied to calculate the cash payout.
> 
> **Minimum Balance for Physical Bullion Redemption**  
> - **10 grams of gold** (or the monetary equivalent based on the prevailing spot price) is the minimum balance required before a holder can request delivery of physical bullion.  
> - Once the 10‑gram threshold is met, the holder may submit a written request and, after the standard processing period (typically 5‑7 business days), receive the physical gold bars or coins.
> 
> ---
> 
> ### Quick Reference
> 
> - **Regulator:** Central Bank of Egypt (CBE)  
> - **Manager:** Al Ahly Capital Holding (Gold Services)  
> - **Early‑Redemption Fees:** 2 % (Y1) → 1.5 % (Y2) → 1 % (Y3) → 0.5 % (post‑Y3)  
> - **Physical‑Bullion Redemption Minimum:** 10 g of gold (or cash equivalent).  
> 
> *Note: The exact fee percentages and minimum balance are those most commonly published in AZ‑Gold’s prospectus and product brochure as of the latest public disclosures. Always verify with the issuing bank or the official AZ‑Gold documentation for any recent updates.*

- *Evaluator Note (RAG):* Answer accurately matches the reference details. (Correctness: 5/5, Relevance: 5/5)
- *Evaluator Note (No-RAG):* Provides entirely incorrect regulator, manager, fee schedule, and redemption minimum. (Correctness: 1/5, Relevance: 5/5)

### Q7 [multi_doc_profile]: I am a university student in Egypt saving EGP 2,000 per month for the next 4 years. Which gold vehicle and strategy should I use, what allocation makes sense, and what mistakes should I avoid?
- **Target File(s):** `20_gold_investor_decision_matrix.txt, 18_egypt_gold_mutual_funds_az_gold_and_regulated_options.txt, 17_egypt_making_charges_masnaeya_and_cashback.txt`
- **Verified Reference:** For a student with a small monthly budget in Egypt and a 4-year horizon, use Monthly Dollar-Cost Averaging (DCA) into an FRA-regulated gold mutual fund like AZ-Gold (avoiding high physical masna'eya and home-storage risk) or save in a money market fund until buying a sealed 8g 21K pound or 5g-10g 24K bar. Target 10% to 15% of long-term savings after an emergency buffer, avoid buying high-masna'eya 1g bars every month or commercial jewelry, and avoid withdrawing from AZ-Gold in Year 1 (~4.9% fee).
- **Top-1 Cosine Sim:** `0.7504`

**With RAG (AurumAI):**
> **Recommended Gold Vehicle & Strategy for a University Student in Egypt**
> 
> | Item | Recommendation | Why it fits your profile |
> |------|----------------|--------------------------|
> | **Vehicle** | **FRA‑regulated gold mutual fund – AZ‑Gold (Azimut Egypt)** | Allows monthly top‑ups as low as EGP 500, avoids the high *masnaʿeya* premium on tiny physical bars, and requires no home‑storage or insurance [Source 1][20_gold_investor_decision_matrix.txt] [Source 3][18_egypt_gold_mutual_funds_az_gold_and_regulated_options.txt] |
> | **Strategy** | **Monthly Dollar‑Cost Averaging (DCA)** – invest the full EGP 2,000 each month (or a consistent portion) into AZ‑Gold via a licensed app (Thndr, Rumble, Mubasher, Azimut) | Smooths out 2026 price volatility and builds disciplined long‑term savings [Source 2][20_gold_investor_decision_matrix.txt] |
> | **Target Allocation** | **10 % – 15 % of your total long‑term savings** (after keeping a small emergency cash buffer) | The decision matrix recommends this range for students and small‑budget savers [Source 1][20_gold_investor_decision_matrix.txt] |
> | **Time Horizon** | **4 years (≈48 months)** – well above the 3‑year minimum for AZ‑Gold to be cost‑effective [Source 3][18_egypt_gold_mutual_funds_az_gold_and_regulated_options.txt] |
> 
> **Key Mistakes to Avoid**
> 
> | Pitfall | Explanation |
> |---------|-------------|
> | **Buying 1‑gram physical bars each month** | The *masnaʿeya* premium (≈ EGP 150‑185 / g) erodes returns dramatically [Source 1][20_gold_investor_decision_matrix.txt] |
> | **Purchasing commercial jewelry as an investment** | Jewelry carries high markup, design risk, and resale uncertainty [Source 1][20_gold_investor_decision_matrix.txt] |
> | **Redeeming AZ‑Gold within the first year** | Early redemption incurs a ~4.9 % fee, which can wipe out a large part of your gains [Source 1][20_gold_investor_decision_matrix.txt] [Source 3][18_egypt_gold_mutual_funds_az_gold_and_regulated_options.txt] |
> | **Skipping the emergency‑cash buffer** | Investing before you have a short‑term liquidity cushion can force a premature, costly withdrawal [Source 1][20_gold_investor_decision_matrix.txt] |
> 
> **Action Plan**
> 
> 1. **Set up an emergency fund** (e.g., 1‑2 months of living expenses) in a liquid account.  
> 2. **Open a regulated account** on a licensed app that offers AZ‑Gold.  
> 3. **Automate a monthly transfer** of EGP 2,000 (or a fixed percentage) into the fund.  
> 4. **Monitor the fund’s performance** but avoid any withdrawal before the 12‑month mark to sidestep the 4.9 % exit fee.  
> 5. **Re‑evaluate annually** to ensure the 10‑15 % allocation still matches your total savings and risk tolerance.
> 
> *Disclaimer: This information is for educational purposes only and does not constitute licensed financial advice.*

**Without RAG (Baseline):**
> **Quick‑look at the numbers**
> 
> | Item | Amount |
> |------|--------|
> | Monthly saving | EGP 2,000 |
> | Horizon | 4 years (48 months) |
> | Total cash saved | EGP 96,000 (≈ US$ 5,200 @ EGP 18.5/USD) |
> | Goal | Preserve buying power and earn a modest real return with gold as a core component |
> 
> ---
> 
> ## 1. Which “gold vehicle” makes sense in Egypt?
> 
> | Vehicle | How you buy it | Typical cost/fees | Liquidity | Suitability for a student |
> |---------|----------------|-------------------|-----------|---------------------------|
> | **Physical 24‑kt gold bars / coins** (e.g., 1 g, 5 g, 10 g, 1 oz) | Purchase from banks, licensed jewelers, or the Egyptian Central Bank’s “Gold Savings” program | Premium 2‑5 % over spot + storage (if you keep at home, no extra fee) | Very high – you can sell to banks/jewelers anytime | Good for “hands‑on” saving, but you must guard against theft & counterfeit |
> | **Gold‑backed exchange‑traded funds (ETFs)** – e.g., **iShares Gold Trust (IAU)**, **SPDR Gold Shares (GLD)** (via an international brokerage) | Buy shares on a foreign exchange (London, NY) through a broker that accepts EGP deposits or a local broker with a “global” platform | Brokerage commission (≈0.1‑0.3 % per trade) + ETF expense ratio (0.12‑0.25 % p.a.) | Very high – can sell any market day | Low storage risk, easier to fractional‑invest, but you need a brokerage that lets you fund with EGP or a USD/EUR account |
> | **Gold‑linked mutual funds / “Gold Savings” accounts** offered by Egyptian banks (e.g., Banque Misr Gold Savings, National Bank of Egypt Gold Account) | Deposit cash; the bank converts it into gold at the daily spot rate and credits you “grams” | Usually a small management fee (≈0.2‑0.5 % p.a.) + spread on conversion | High – you can request redemption in cash or gold | Very convenient, no need for a foreign broker, but you are exposed to the bank’s credit risk |
> | **Digital gold platforms** (e.g., **Goldmoney**, **BullionVault**, **e‑Gold** apps) | Mobile app; you buy fractional grams that are stored in professional vaults | Platform fee (≈0.1‑0.3 % p.a.) + transaction fee (≈0.5‑1 % per purchase) | High – you can sell any day, cash out to local bank | Good for tech‑savvy students, but you need a verified account and sometimes a foreign bank link |
> 
> **Bottom line:** For a student with modest monthly cash, the **simplest and cheapest** route is usually the **bank‑run Gold Savings account** or **physical small‑denomination gold coins/bars** bought from a reputable bank or licensed jeweler. If you already have a brokerage that lets you fund in USD/EUR, an **ETF** is the next‑best low‑cost, fully liquid option.
> 
> ---
> 
> ## 2. Suggested allocation (48 months)
> 
> | Asset | % of total portfolio | Reason |
> |-------|---------------------|--------|
> | **Gold (core)** | **55 %** (≈ EGP 52,800) | Preserves value against inflation & EGP devaluation; serves as “safety net”. |
> | **Cash / short‑term savings** | **20 %** (≈ EGP 19,200) | Keeps liquidity for tuition, emergencies, and to fund the next gold purchase without selling at a low price. |
> | **Diversified equities / index fund** | **15 %** (≈ EGP 14,400) | Adds growth potential; a global low‑cost index fund (e.g., MSCI World ETF) can be bought via the same brokerage used for gold ETFs. |
> | **Fixed‑income / government bonds** | **10 %** (≈ EGP 9,600) | Provides a modest, stable return and reduces overall volatility. Egyptian Treasury bonds or a short‑term money‑market fund work. |
> 
> **How to implement month‑by‑month**
> 
> 1. **Month 1‑12** – Build the cash buffer first. Deposit the full EGP 2,000 each month into a high‑interest savings account (≈ 3‑4 % p.a.

- *Evaluator Note (RAG):* Provides the same vehicle, strategy, allocation range, and mistake warnings as the reference. (Correctness: 5/5, Relevance: 5/5)
- *Evaluator Note (No-RAG):* Provides a different strategy and allocation than the reference, missing key specifics. (Correctness: 2/5, Relevance: 4/5)

### Q8 [multi_doc_profile]: I have EGP 150,000 needed for tuition and wedding expenses in 6 months. Should I put this money into physical gold bars or AZ-Gold during 2026 to hedge inflation?
- **Target File(s):** `20_gold_investor_decision_matrix.txt, 09_rebalancing_and_investment_horizon_for_gold.txt, 12_bid_ask_spreads_and_hidden_transaction_costs.txt`
- **Verified Reference:** No — short-term cash needed within 12 months should have a 0% gold allocation. Gold requires a 3 to 5+ year horizon because it can suffer 8% to 15% short-term pullbacks, physical bullion carries a 2% to 6% spread and making-charge hurdle, and AZ-Gold charges a ~4.9% Year-1 redemption fee. Keep the EGP 150,000 in liquid money market funds, short-term Treasury bills (T-bills), or high-yield savings instead.
- **Top-1 Cosine Sim:** `0.6797`

**With RAG (AurumAI):**
> Based on the decision‑matrix guidance for investors who need cash within 12 months, **gold is not recommended** for your situation.
> 
> - **Profile 4 – Short‑Term Cash Holder (Horizon < 12 months)** states that the primary goal is capital preservation with 100 % nominal certainty and immediate liquidity.  
> - The matrix advises a **0 % gold allocation** for funds needed in six months, because gold typically experiences 8 %–15 % short‑term pullbacks, carries a 2 %–6 % making‑charge and spread, and AZ‑Gold imposes a ~4.9 % early‑redemption fee in the first year.  
> - Instead, keep the EGP 150,000 in **liquid money‑market funds, short‑term Treasury bills, or a high‑yield savings account** to preserve purchasing power and ensure the money is available when tuition and wedding expenses arise.  
> 
> So, for a six‑month horizon, neither physical gold bars nor AZ‑Gold are suitable hedges; retain the cash in a short‑term, low‑risk liquid vehicle.
> 
> *Disclaimer: This information is for educational purposes only and does not constitute licensed financial advice.*

**Without RAG (Baseline):**
> **Short answer:** For a 6‑month horizon, physical gold bars are generally a **less suitable hedge** than a liquid, low‑cost gold‑linked product such as an exchange‑traded fund (e.g., “AZ‑Gold”).  
> 
> ---
> 
> ### Why a 6‑month horizon matters
> | Factor | Physical gold bars | AZ‑Gold (gold‑ETF/ETN) |
> |--------|-------------------|------------------------|
> | **Liquidity** | Must find a buyer, store‑to‑store, possible premiums/discounts; can take days‑weeks. | Traded on the exchange; you can sell any market day at the prevailing NAV/price. |
> | **Transaction costs** | Buying premium + storage + insurance + possible sales discount. | Small brokerage commission (often 0.1‑0.2 %); no storage fees. |
> | **Price tracking** | Spot price + dealer markup; may deviate from market price. | Tracks spot price very closely (tracking error < 0.2 %). |
> | **Safety / custody** | Risk of theft, loss, or damage; you must arrange secure storage. | Custodied by the fund; regulatory oversight. |
> | **Tax treatment (Egypt)** | Capital‑gain tax may apply on sale; storage fees are not deductible. | Same tax on gains, but you can offset with other securities; easier record‑keeping. |
> | **Inflation hedge** | Works over **years**, not months; short‑term price swings dominate. | Same long‑term hedge, but you can exit quickly if inflation expectations change. |
> 
> ### Inflation outlook for 2026 (Egypt)
> * Egyptian inflation has been volatile (double‑digit in recent years).  
> * Short‑term drivers (food, fuel subsidies, fiscal policy) can cause **monthly swings of ±2‑3 %** in the local‑currency price of gold.  
> * Over **six months**, the expected real return on gold is close to **zero** after accounting for transaction costs.
> 
> ### Practical recommendation
> 1. **Keep the cash in a high‑yield, low‑risk vehicle** for the next 6 months (e.g., a short‑term Egyptian‑bank term deposit or a money‑market fund) to preserve capital and earn a modest nominal return (≈ 8‑12 % p.a. in Egypt today).  
> 2. **Allocate a small portion (≤ 10 % of the 150 k EGP)** to a liquid gold‑linked product like AZ‑Gold **only if you want a modest inflation hedge** and are comfortable with the market risk.  
> 3. **Avoid committing the full amount to physical bars** unless you have a strong need for physical possession and are prepared to bear storage, insurance, and liquidity costs.
> 
> ### Quick action checklist
> | Step | What to do |
> |------|------------|
> | 1 | Open a brokerage account that offers AZ‑Gold (or a similar gold‑ETF). |
> | 2 | Deposit up to 15,000 EGP (10 % of the total) into the ETF; keep the rest in a short‑term term deposit. |
> | 3 | Set a stop‑loss or a target exit date (e.g., 5 months) to ensure you have the cash when tuition/wedding bills arrive. |
> | 4 | Review inflation data and the Egyptian pound’s exchange rate monthly; adjust the allocation if inflation expectations shift dramatically. |
> 
> **Bottom line:** For a six‑month need, a liquid gold‑ETF like AZ‑Gold is far more practical than physical bars, but the hedge benefit is modest. Preserve most of the money in a low‑risk, interest‑bearing account and only allocate a small slice to gold if you want some protection against a sudden inflation spike.

- *Evaluator Note (RAG):* Accurately mirrors reference answer and directly addresses the query. (Correctness: 5/5, Relevance: 5/5)
- *Evaluator Note (No-RAG):* Provides advice contrary to the reference recommendation of no gold allocation. (Correctness: 2/5, Relevance: 5/5)

### Q9 [out_of_scope]: What are the best Ethereum Layer-2 staking pools in 2026, and what APY can I earn from Bitcoin DeFi yield farming?
- **Target File(s):** `None (Out of Scope)`
- **Verified Reference:** Out of scope: AurumAI must explicitly refuse and state that its gold-investing knowledge base does not cover cryptocurrency, Ethereum staking, or Bitcoin yield farming.
- **Top-1 Cosine Sim:** `0.3491`

**With RAG (AurumAI):**
> I’m sorry, but the AurumAI knowledge base does not contain information on Ethereum Layer‑2 staking pools or Bitcoin DeFi yield farming. Therefore I cannot provide an answer to that request.
> 
> *Disclaimer: This information is for educational purposes only and does not constitute licensed financial advice.*

**Without RAG (Baseline):**
> I’m sorry, but I don’t have information about specific Ethereum Layer‑2 staking pools or Bitcoin‑DeFi yield‑farming rates for the year 2026. My training only includes data up to 2024, and I can’t forecast future products or APYs. 
> 
> If you’re looking for up‑to‑date options, I recommend:
> 
> 1. **Checking the official sites and dashboards** of the major Layer‑2 networks (e.g., Arbitrum, Optimism, zkSync, StarkNet). They usually list supported validators or staking‑as‑a‑service providers and the current rewards.
> 2. **Visiting reputable DeFi aggregators** such as **DefiLlama**, **Yieldwatch**, or **Zapper** for real‑time APY data on Bitcoin‑related yield farms (e.g., wrapped BTC on Aave, Compound, or on cross‑chain platforms like Thorchain, RenVM, and Layer‑2 bridges).
> 3. **Reading community forums** (Reddit r/ethstaker, r/defi, Discord channels) where users share the latest pool performance and any recent changes in incentives.
> 4. **Being cautious of risk**—high APYs often come with higher smart‑contract, liquidity, or market risk. Look for pools that have undergone audits and have a solid track record.
> 
> If you have a specific Layer‑2 network or a particular Bitcoin‑wrapped token (e.g., wBTC, renBTC, tBTC) in mind, I can give you a more detailed overview of the typical staking or yield‑farming mechanisms that were popular up to 2024.

- *Evaluator Note (RAG):* The answer properly refuses, matching the out-of-scope policy. (Correctness: 5/5, Relevance: 5/5)
- *Evaluator Note (No-RAG):* Provided partial info instead of proper refusal, thus low correctness and relevance. (Correctness: 2/5, Relevance: 2/5)

### Q10 [out_of_scope]: What are the current Central Bank of Egypt residential real estate mortgage interest rates and down-payment tiers for buying an apartment in New Cairo?
- **Target File(s):** `None (Out of Scope)`
- **Verified Reference:** Out of scope: AurumAI must explicitly refuse and state that its gold-investing knowledge base does not cover real estate mortgages or Egyptian housing finance tiers.
- **Top-1 Cosine Sim:** `0.5185`

**With RAG (AurumAI):**
> I’m sorry, but the AurumAI knowledge base does not contain information on Central Bank of Egypt residential real‑estate mortgage interest rates or down‑payment tiers. This request is outside the scope of gold‑investment guidance.
> 
> *Disclaimer: This information is for educational purposes only and does not constitute licensed financial advice.*

**Without RAG (Baseline):**
> I’m sorry, but I don’t have access to real‑time data, so I can’t provide the current Central Bank of Egypt mortgage rates or the specific down‑payment tiers for apartments in New Cairo. For the most up‑to‑date information, I recommend checking:
> 
> 1. **Central Bank of Egypt (CBE) website** – they publish the latest benchmark lending rates and any official guidelines for mortgage financing.  
> 2. **Major Egyptian banks** (e.g., National Bank of Egypt, Banque Misr, Commercial International Bank, Housing and Development Bank) – their mortgage product pages often list the current interest rates, loan‑to‑value (LTV) ratios, and required down‑payment percentages.  
> 3. **Real‑estate portals** (such as Property Finder Egypt, OLX Egypt, or local developer websites) – they sometimes include financing calculators that reflect the prevailing rates and down‑payment structures for projects in New Cairo.  
> 4. **A real‑estate agent or mortgage broker** – they can give you a quick snapshot of the typical rates and down‑payment requirements for the specific development you’re interested in.
> 
> Generally, Egyptian mortgage products fall into a few common down‑payment tiers:
> 
> | Down‑payment (LTV) | Typical Mortgage Rate* |
> |--------------------|------------------------|
> | 20 % (80 % LTV)     | 9 %–12 % per annum     |
> | 30 % (70 % LTV)     | 8 %–11 % per annum     |
> | 40 % (60 % LTV)     | 7 %–10 % per annum     |
> 
> \*These ranges are illustrative based on recent historical data; actual rates can vary by bank, borrower credit profile, loan term, and whether the loan is tied to a specific developer’s financing program.
> 
> **Next steps:**  
> - Visit the CBE’s “Monetary Policy” or “Statistical Data” sections for the latest benchmark rate.  
> - Look up the mortgage sections of the banks mentioned above for their current product sheets.  
> - Contact a local mortgage specialist to confirm the exact down‑payment requirements for the specific New Cairo apartment you’re considering.  
> 
> If you need help interpreting the information you find or comparing different offers, feel free to share the details and I can walk you through the numbers.

- *Evaluator Note (RAG):* Properly refused as out-of-scope without providing information. (Correctness: 5/5, Relevance: 5/5)
- *Evaluator Note (No-RAG):* Failed to refuse out-of-scope request and provided speculative information. (Correctness: 1/5, Relevance: 2/5)
