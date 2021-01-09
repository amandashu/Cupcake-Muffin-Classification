## Data Collection
The data is scraped from baking blogs [Baking Bites](https://bakingbites.com/), [Sally's Baking Addition](https://sallysbakingaddiction.com/), and [The Baker Chick](https://www.thebakerchick.com/), implemented in this [file](https://github.com/amandashu/Cupcake-Muffin-Classification/blob/main/src/data/scrape.py). It scrapes both cupcake and muffin recipes from these sites into `/data/recipes.csv`, which is a csv file containing columns `link`, `type` (cupcake or muffin), and `ingredients` (list of each ingredient line). When possible, only the main part of the recipe is scraped (i.e. the cake portion of cupcakes and not the frosting).

Example row of scraped data:

| link | type | ingredients
|------|------|------|
| https://sallysbakingaddiction.com/pumpkin-crum... | muffin | ['1 and 3/4 cups (220g) all-purpose flour (spoon & leveled)','1 teaspoon baking soda','2 teaspoons ground cinnamon','1 teaspoon pumpkin pie spice','1/2 teaspoon salt','1/2 cup (120ml) canola or vegetable oil','1/2 cup (100g) granulated sugar','1/2 cup (100g) packed light or dark brown sugar', '1 and 1/2 cups (340g) canned pumpkin puree','2 large eggs, at room temperature','1/4 cup (60ml) milk, at room temperature']


## Data Cleaning and Preparation
Note many ingredients that are scraped are not common among all recipes (i.e pumpkin puree). I limit the ingredients to look at to only flour, baking powder (or baking soda), butter (or oil), sugar, egg, and vanilla,  and did a one hot encoding. In some cases column might contain a list of ingredient lines (i.e. when a recipe has both baking powder and baking soda). The first intermediary data frame looks like this:

| link | type | flour | baking powder/soda | butter/oil | sugar | egg | vanilla
|------|------|------|------|------|------|------|------|
| https://www.thebakerchick.com/strawberry-lemon... | cupcake | 3 cups all-purpose flour | [1 tablespoon baking powder] | ['1 cup (2 sticks) unsalted butter, room temperature'] | [2 cups sugar] | 4 large eggs, room temperature | 1 teaspoon vanilla extract

The next step is to clean each column from strings/lists to a single float value representing how much of a ingredient is in that recipe (i.e. 2 in flour column means 2 cups of flour). As with most text data written by humans, there are many things to consider when cleaning these ingredient columns.

**Flour**: This column is normalized to cups of flour. Here are several problems noted when cleaning this column:

1. Different ways to write the amounts, including special characters for fractions and using spaces, no spaces, or dashes. (i.e. ½, 1½, 1 ½, 1 1/2, 1-1/2)
2. Use of tablespoon or teaspoon measurement instead of cups. These need to be converted (1 tablespoon = 0.0625 cups, 1 teaspoon = 0.0208333 cups)
3. Use of abbreviations versus full names. (i.e. tbsp vs tablespoon, tsp vs teaspoon)
4. When using multiple types of measurements, using different conjunctions. (i.e. plus, +, and)

Here are example inputs and outputs to the cleaning function for flour.

| Original String | Cleaned |
|------|------|
| ¾ cup sugar | 0.75|
| 1½ cups all purpose flour | 1.5|
| 1 1/2 cups all-purpose flour	| 1.5|
| 1/2 cup brown sugar| 0.5 |
| 3 cups all-purpose flour | 3|
| 2-1/2 cups all-purpose flour | 2.5|
| 1 and 1/2 cups (300g) granulated sugar, divided| 1.5|
| 3 tablespoons flour | 0.1875|
| 14 tbsp all purpose flour (1 cup minus 2 tbsp) | 0.875|
| 1 cup plus 2 tablespoons flour | 1.125|
| 2 tablespoons and 1 teaspoon white sugar | 0.1458333|
| 1 cup + 2 tablespoons all-purpose flour | 1.125|
| 1 cup + 2 tbsp sugar | 1.125

**Baking Powder and Baking Soda**: The baking powder/soda column is converted to teaspoons of baking powder. In addition to #1 and #3 listed under flour, here are other problems to consider:

1. Use of tablespoon measurement. These need to be converted (1 tablespoon = 3 teaspoons)
2. Baking soda measurement is converted to baking powder (1/4 teaspoon baking soda = 1 teaspoon of baking powder)
3. For recipes with both powder and soda, these are added together into baking powder equivalent.

| Original String | Cleaned |
|------|------|
| \[2 tsp baking powder\]| 2|
| \[2½ tsp. baking powder\]| 2.5 |
| \[1 tablespoon baking powder\] | 3|
| \[1 tbsp. baking powder\]| 3|
| \[1 teaspoon baking soda\] | 4 |
| \[1/8 tsp baking soda\] | 0.5
| \['¾ teaspoons baking soda', '¼ teaspoon baking powder'\]| 3.25|
| \['1-1/2 teaspoons baking powder', '¼ teaspoon baking soda'\]| 2.5 |
| \['1 1/2 teaspoons baking soda', '3/4 teaspoon baking powder'\] |6.75 |
| \['1 ½ tsp. baking powder', '¼ tsp. baking soda'\] | 2.5|

**Oil and Butter**: The oil/butter column is converted to cups of butter. In addition to #1 and #3 listed under flour, here are other problems to consider:

1. Use of measurements such as sticks, ounces, or tablespoons instead of cups. These need to be converted (1 cup butter = 2 sticks butter, 1 cup butter = 16 tablespoons butter or oil, 1 cup butter = 8 ounces butter).
2. Oil is converted to butter equivalent (use 3/4 amount of oil for butter)
3. For recipes with both oil and butter, these are added together into butter equivalent.

| Original String | Cleaned |
|------|------|
|\[1 stick of butter, at room temperature\] |0.5 |
|\[1/2 cup (120ml) canola or vegetable oil\] | 0.667|
|\[1½ tablespoons vegetable oil\]|0.125 |
|\[5 tbsp butter, melted and cooled\]| 0.3125 |
|\[¼ cup plus 2 tablespoons vegetable oil\]|0.5|
|\[8 ounces (2 sticks) unsalted butter, softened\]|1|
| \['3 tablespoons canola/vegetable oil', '2 sticks unsalted butter- softened'\]| 1.25 |

**Sugar**: The sugar column is also converted to cups. In addition to #1 - #3 listed under flour, other thing to consider are:
1. Use of sugar substitutes such as honey or syrup. These are converted with 1 to 1 ratio.
2. Having optional ingredients. These are not included

| Original String | Cleaned |
|------|------|
| ¾ cup sugar | 0.75|
| 1/2 cup brown sugar| 0.5 |
| 1 and 1/2 cups (300g) granulated sugar, divided| 1.5|
| 2 tablespoons and 1 teaspoon white sugar | 0.1458333|
| 1 cup + 2 tbsp sugar | 1.125|
| \['1/4 cup honey or pure maple syrup', '1/2 cup light or dark brown sugar, loosely packed'\] | 0.75 |
| \['1/3 cup (80ml) pure maple syrup', 'optional: 3 tablespoons oats and coconut sugar for sprinkling'\] | 0.333

**Egg**: The eggs column is converted to number of eggs. Here are things to consider:

1. Use of tablespoon or cup measurement. These need to be converted (1 cup = 4 eggs, 1 teaspoon = 0.25 egg)
2. Use of 'or'

| Original String | Cleaned |
|------|------|
| 1 egg | 1|
| 4 large eggs, room temperature| 4|
| 1/3 cup (80ml) pasteurized egg whites, at room temperature | 1.333|
| 2 large egg whites or 1 large egg| 1|
| 2 tbsp egg (1/2 large, beaten egg)| 0.5|

**Vanilla**: The vanilla column is converted to teaspoons of vanilla extract. In addition to #1 and #3 listed under flour and #1 listed under baking powder/soda, another problem to consider is:

1. Use of vanilla bean instead of extract. These need to be converted (1 vanilla bean = 1 teaspoon extract)

| Original String | Cleaned |
|------|------|
| 1 teaspoon vanilla extract | 1|
| ½ teaspoon pure vanilla extract | 0.5|
| 1/8 teaspoon of vanilla extract| 0.128|
| 1 1/2 tsp vanilla extract	| 1.5 |
| 2 and 1/2 teaspoons pure vanilla extract| 2.5|
| 2 tsp vanilla extract or vanilla bean paste | 2|
| 1 tbsp vanilla extract | 3|
| 2 tablespoons pure vanilla extract| 6|
| 1 vanilla bean, split lengthwise | 1|


The next intermediary data frame looks like this:

| link | type | Flour (cups) | Baking Powder (tsp) | Butter (cups) | Sugar (cups) | Eggs | Vanilla (tsp)
|------|------|------|------|------|------|------|------|
| https://www.thebakerchick.com/strawberry-lemon... | cupcake | 3 | 3 | 1 | 2 | 4 | 1

Since the recipes might produce a different number of cupcakes/muffins, rather than actual amounts of each ingredient, these numbers are converted to percentages (i.e. 40 in flour column indicates flour makes up 40% of all the ingredients). Note that a few columns are first converted to cups in order to make these calculation. A column for which blog the recipe comes from is also added.

The final clean data frame, which is outputted to `/data/recipes_clean.csv`, looks like this:

| link | blog | type | Flour (%) | Baking Powder (%) | Butter (%) | Sugar (%) | Eggs (%) | Vanilla (%)
|------|------|------|------|------|------|------|------|------|
| https://www.thebakerchick.com/strawberry-lemon... | The Baker Chick | cupcake | 42.352941	 | 0.882353 | 14.117647	| 28.235294	| 14.117647	| 0.294118

## Notes
It is possible that there are cases where not all the intended ingredients were scraped or partially scraped, since the scraping script makes assumptions about the structure of each webpage and there may be outlier pages that do not conform. The only recipes that were dropped from the data are recipes with no flour, as they may link to another recipe for the cake or use non traditional ingredients like wafers or oats.
