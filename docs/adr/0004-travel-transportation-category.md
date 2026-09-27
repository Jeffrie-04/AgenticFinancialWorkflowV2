# ADR 0004: Travel/Transportation category

Context

The v1 categorizer has four purchase categories: Utilities, Shopping, Dining and Other. Flights, hotels, rideshares, parking, tolls and fuel all land in Other, alongside shipping, bank fees and anything else that fits nowhere. For a business that travels (like the consultant demo business), Other becomes one of the largest categories and tells the owner nothing about where the money went.

I believed a dedicated travel category would be more useful, but I hadn't measured it. So I added it as a new prompt version (v2) and compared v1 and v2 on the same hand-labeled gold set, rather than changing production directly.

Decision
Name and definition

The category is Travel/Transportation. It covers:

hotels (including Airbnb and hostels)
airfare
rideshare (Uber, Lyft)
rental cars
public transit
tolls
parking
gasoline
EV charging
Defined by what was bought, not what it's used for

The rule is based on the purchase itself. Any gasoline purchase is Travel/Transportation, whatever it powers: a car, a work truck, or a landscaper's mowers and equipment. I chose this because a simple, checkable rule can be applied the same way by the model and by me when labeling. "It depends on what the fuel was used for" can't be decided from a single transaction line.

Discretionary

Travel/Transportation counts as discretionary spend in the fixed-vs-discretionary KPI. Travel can usually be cut back or rescheduled, unlike rent, insurance or software subscriptions. This also matches v1, where these purchases were in Other, which was already discretionary, so the existing KPIs don't shift.

v1 is frozen

The category is added to the Category enum, but each prompt version keeps its own explicit list of allowed categories. v1's list stays exactly Utilities, Shopping, Dining, Other. Without this, adding the enum value would have silently changed the v1 prompt, and v1 would have started accepting Travel/Transportation, which would make the v1-vs-v2 comparison meaningless. Production stays on v1 until the eval supports a switch.

Wording fixes made before seeing any results

Two ambiguities were fixed in the v2 prompt before labeling and before running the eval:

"gas" became "gasoline": Utilities already lists "gas" (the natural gas bill), so "gas" in the travel line could be read either way.
"rental cars, public transit" were added: they're clearly transportation, but weren't in the original list. Labeling them Travel/Transportation while the prompt didn't mention them would have graded v2 against a rule it was never given.

Changing the prompt after seeing the scores would fit it to my own test set and inflate the v2 result. Fixing ambiguities that were visible beforehand keeps the comparison fair. After these two changes, the definitions were frozen for the eval.

Consequences

Positive

Owners see travel spend as its own line instead of hidden in Other.
The value of the change is measured on a hand-labeled gold set, not assumed.
v1 stays reproducible, so the comparison is fair.

Trade-offs

For the landscaper, equipment fuel is really an operating cost, yet it's counted as Travel/Transportation. That's the cost of a rule based on what was bought.
Scoring v1 requires mapping gold Travel/Transportation labels to Other, since v1 can't produce the new category.
A second category now has "gas" in its guide line (Utilities' natural gas bill). The "gasoline" wording separates them, and the gold set includes a natural-gas-bill row to test it.
Future work

Per-business category rules. Some businesses would categorize the same purchase differently. For a landscaper, fuel for equipment could go to an "Operations" category rather than Travel/Transportation. The project already configures other behavior per business (sign_convention, date_format in SOURCE_CONFIGS), so per-business category overrides could reuse that pattern, possibly as deterministic rules applied before the model.
