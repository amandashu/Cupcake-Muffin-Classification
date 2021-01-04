### Flour and Sugar
1. Different ways to write the amounts, including special characters for fractions and using spaces, no spaces, or dashes. (i.e. 1½, 1 1/2, 1-1/2)
2. Use of tablespoon or teaspoon measurement instead of cups. These need to be converted (1 tablespoon = 0.0625 cups, 1 teaspoon = 0.0208333 cups)
3. Use of abbreviations versus full names. (i.e. tbsp vs tablespoon)
4. When using multiple types of measurements, using different conjunctions. (i.e. plus, +, and)


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


### Egg
1. Use of tablespoon or cup measurement. These need to be converted (1 cup = 4 eggs, 1 teaspoon = 0.25 egg)
2. Use of 'or'

| Original String | Cleaned |
|------|------|
| 1 egg | 1|
| 4 large eggs, room temperature| 4|
| 1/3 cup (80ml) pasteurized egg whites, at room temperature | 1.333|
| 2 large egg whites or 1 large egg| 1|
| 2 tbsp egg (1/2 large, beaten egg)| 0.5|

### Vanilla
1. Different ways to write the amounts, including special characters for fractions (i.e. ½, 1/8, and)
2. Use of tablespoon measurement. These need to be converted (1 tablespoon = 3 teaspoons)
3. Use of abbreviations versus full names. (i.e. tsp vs teaspoon)
4. Use of vanilla bean instead of extract. These need to be converted (1 vanilla bean = 1 teaspoon extract)


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


### Baking Powder and baking Soda
1. Different ways to write the amounts, including special characters for fractions (i.e. ½, 1 1/2, 1-1/2, 1 ½, 2½)
2. Use of tablespoon measurement. These need to be converted (1 tablespoon = 3 teaspoons)
3. Use of abbreviations versus full names. (i.e. tsp vs teaspoon)
4. Baking soda measurement is converted to baking powder (1/4 teaspoon baking soda = 1 teaspoon of baking powder)
5. For recipes with both powder and soda, these are added together into baking powder equivalent.


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



### Oil and Butter
1. Different ways to write the amounts, including special characters for fractions (i.e. ½, 1/2)
2. Use of measurements such as sticks, ounces, or tablespoons instead of cups. These need to be converted (1 cup butter = 2 sticks butter, 1 cup butter = 16 tablespoons butter or oil, 1 cup butter = 8 ounces butter).
3. Use of abbreviations versus full names. (i.e. tsp vs teaspoon)
4. Oil is converted to butter equivalent (use 3/4 amount of oil for butter)
5. For recipes with both oil and butter, these are added together into butter equivalent.


| Original String | Cleaned |
|------|------|
|\[1 stick of butter, at room temperature\] |0.5 |
|\[1/2 cup (120ml) canola or vegetable oil\] | 0.667|
|\[1½ tablespoons vegetable oil\]|0.125 |
|\[5 tbsp butter, melted and cooled\]| 0.3125 |
|\[¼ cup plus 2 tablespoons vegetable oil\]|0.5|
|\[8 ounces (2 sticks) unsalted butter, softened\]|1|
| \['3 tablespoons canola/vegetable oil', '2 sticks unsalted butter- softened'\]| 1.25 |
