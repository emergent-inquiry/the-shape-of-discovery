# literature-scout

Search for academic papers, verify patent data, and validate prior art claims.

## When to Use

- When adding new breakthroughs to `data/breakthroughs/*.json` — verify patent numbers and filing dates
- When a new confound is identified — search for related methodology literature
- When writing or updating README.md — check for new TDA-on-networks papers
- Periodically — verify the "gap in the literature" claim (no prior persistent homology on patent citation networks)

## Verifying Breakthrough Patents

1. Use web search to find the patent on Google Patents or USPTO PAIR
2. Confirm: patent number, title, filing date, grant date, assignee, CPC classes
3. Cross-check against our data: does this `patent_id` exist in `data/patents.parquet`?
4. Verify breakthrough significance: is this widely recognized as a breakthrough in the literature?
5. Check that CPC section assignments match what we claim in the JSON catalog under `data/breakthroughs/`

## Searching for Prior Art (TDA on Patents)

This is the most important search. Our contribution claims that nobody has applied persistent homology to the patent citation network. If this is wrong, the project's novelty claim collapses.

1. Search for: "persistent homology patent", "topological data analysis patent network", "Betti numbers citation network", "TDA innovation network", "persistent homology knowledge graph"
2. Check Google Scholar, arXiv, Semantic Scholar
3. Distinguish carefully:
   - **Nakamura et al. (2023)** uses TDA Mapper (NOT persistent homology) — different technique, not prior art for our specific method
   - Papers applying TDA to **other** networks (social, biological) are related work, not direct prior art
   - Papers using network metrics (not topology) on patents are context, not competition
   - Papers applying persistent homology to **other** innovation data (academic citations, R&D networks) are close — assess carefully
4. **If a paper IS found** that applies persistent homology to patent citations: this is critical — report immediately so CLAUDE.md prior art section can be updated

## Searching for Confound Literature

For each confound in CONFOUNDS.md, search for methodology papers that discuss the issue and potential mitigations:

| Confound | Search Terms |
|----------|-------------|
| #1 Examiner citations | "examiner citation bias patents", "applicant vs examiner citations USPTO" |
| #2 Prosecution lag | "patent prosecution delay", "grant lag citation timing" |
| #3 Policy shocks | "Alice Corp patent impact", "America Invents Act citation effects" |
| #4 Strategic patenting | "patent thicket detection", "defensive patenting network" |
| #5 Citation cultures | "citation norms technology fields", "patent citation practices by domain" |
| #6 Survivorship | "survivorship bias patent data", "rejected patent applications data" |
| #7 CPC changes | "patent classification reclassification", "CPC IPC migration effects" |
| #8 Self-citation | "assignee self-citation patents", "corporate self-citation network" |
| #9 Truncation | "citation truncation bias bibliometrics", "citation window bias" |

## Output Format

For each search, report:
- **Queries used**: exact search terms
- **Findings**: author, year, title, URL/DOI
- **Relevance**: direct prior art / related work / methodology reference / not relevant
- **Action items**: e.g., "patent US8697359 confirmed: CRISPR, filed 2012-10-19, CPC C12N15/10"

## Important

- Always note the date of the search (results change over time)
- The gap-in-literature claim in CLAUDE.md is dated March 2026 — if new work is found, it must be updated
- Accuracy matters more than speed for patent number verification
- When verifying patents, prefer primary sources (USPTO, Google Patents) over secondary summaries
- Known prior art to always acknowledge: Érdi et al. (2013), Mariani et al. (2018), Nakamura et al. (2023)
