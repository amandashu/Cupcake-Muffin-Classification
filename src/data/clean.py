import pandas as pd
import unicodedata
import re

def first_num(x):
    """
    Takes in string and outputs number, taking into account special characters
    and other ways of denoting numbers
    """
    if '/' in x:
        space_ind = x.find(' ')
        if space_ind ==-1:
            num = eval(x)
        else:
            num = float(x[:space_ind]) + eval(x[space_ind+1:])
    else:
        if len(x)==1:
            try:
                num = int(x)
            except:
                num = unicodedata.numeric(x)
        else:
            try:
                num = int(x)
            except:
                num = int(x[:-1]) + unicodedata.numeric(x[-1])
    return num

def create_blog(link):
    if 'bakerchick' in link:
        return 'The Baker Chick'
    if 'sally' in link:
        return "Sally's Baking Addiction"
    if 'bakingbites' in link:
        return 'Baking Bites'

def convert_flour_sugar(line):
    """
    Converts flour and sugar ingredient lines to a number (in cups)
    """
    if pd.isna(line):
        return 0
    if 'optional' in line:
        return 0
    line = re.sub(r'(and)|-|(plus)|\+',' ', line)
    if line.find('(') != -1:
        line = line[:line.find('(')] + line[line.find(')')+1:]

    line = line.replace('and','')

    tsp_ind = line.find('cup')
    x = line[:tsp_ind-1]

    if 'tablespoon' in line or 'tbsp' in line:
        num = int(line[:line.find(' ')]) # first number
        if 'cup' in line:
            cup = int(line[:line.find(' ')])
            tbsp = line[line.find('cup')+3:].strip()
            tbsp = int(tbsp[:tbsp.find(' ')])*0.0625
            num = cup + tbsp
        elif 'teaspoon' in line:
            tspace_ind = line[:line.find('teaspoon')]
            num = num*0.0625 + int(tspace_ind[-2])*0.0208333
        else: # only tablespoons
            num = num*0.0625
        return num

    num = first_num(x)
    return num

def convert_egg(line):
    """
    Converts egg ingredient lines to a number
    """
    if pd.isna(line):
        return 0
    if 'eggnog' in line:
        return 0
    if 'tbsp' in line:
        return 1/2
    if 'cup' in line:
        cup = line[:line.find(' cup')]
        return eval(cup)*4
    if 'or'in line:
        or_ind = line.find('or')
        return int(line[or_ind+3])
    if len(re.findall('egg', line)) == 2:
        space_ind = line.find(' + ')
        return int(line[0]) + int(line[space_ind+3])
    return int(line[0])

def convert_van(line):
    """
    Converts vanilla ingredient lines to number (in teaspoons)
    """
    if pd.isna(line):
        return 0

    line = line.replace('teaspoon','tsp')
    line = line.replace('and','')

    tsp_ind = line.find('tsp')
    x = line[:tsp_ind-1]

    if 'tablespoon' in line or 'tbsp' in line:
        return int(line[0])*3
    if 'bean' in line:
        return int(line[0])

    num = first_num(x)
    return num

def convert_bpbs(lst):
    """
    Converts baking powder/soda ingredient lines to number (in teaspoons in baking powder)
    """
    if type(lst) != list:
            return 0
    tot = 0
    for line in lst:
        line = line.replace('teaspoon','tsp')
        line = line.replace('and','')
        line = line.replace('-', ' ')
        line= line.replace('â','')

        t_ind = line.find('t')
        x = line[:t_ind-1]

        num = first_num(x)

        if 'tablespoon' in line or 'tbsp' in line:
            num = num*3
        if 'soda' in line:
            num = num*4

        tot += num
    return tot

def convert_bo(lst):
    """
    Converts butter/oil ingredient lines to number (in cups of butter)
    """
    if type(lst) != list:
            return 0
    tot = 0
    for line in lst:
        # delete parenthesis
        line = line[:line.find('(')] + line[line.find(')')+1:]
        line = line.replace('â','')

        s_ind = line.find(' ')
        x = line[:s_ind]

        num = first_num(x)

        if "confectioners\' sugar" in line:
            num = 1
        if 'tablespoon' in line or 'tbsp' in line:
            if 'cup' in line:
                tbsp = int(line[line.find('tablespoon')-2])
                num = num + tbsp/16
            else:
                num = num/16
        if 'stick' in line:
            num = num/2
        if 'ounce' in line:
            num = num/8
        if 'oil' in line:
            num = num*4/3
        tot += num

    return tot

def has_ingredient(lst, ing):
    """
    Checks list if ingredient exists
    """
    for x in lst:
        if ing in x:
            return True
    return False

def one_hot(lst):
    """
    Takes in list of ingredients and output series of the 6 main ingredients
    """
    dct = {}
    butter = 0
    for x in lst:
        if 'frosting' in x:
            break
        if 'flour' in x and 'flour' not in dct.keys():
            dct['flour'] = x
        if 'sugar' in x and 'sugar' not in dct.keys():
            dct['sugar'] = x
        if 'egg' in x and 'egg' not in dct.keys():
            dct['egg'] = x
        if ('baking powder' in x or 'baking soda' in x):
            if 'baking powder/soda' in dct.keys():
                dct['baking powder/soda'].append(x)
            else:
                dct['baking powder/soda'] = [x]
        if ('vanilla extract' in x or 'bean' in x) and 'vanilla' not in dct.keys():
            dct['vanilla'] = x
        if ('butter' in x or ' oil' in x) and 'buttermilk' not in x and 'pumpkin butter' not in x and 'butter extract' not in x and 'buttercream' not in x and 'butterscotch' not in x and 'butternut' not in x and 'peanut butter' not in x and 'cookie butter' not in x:
            if 'butter/oil' not in dct.keys():
                if 'butter' in x:
                    butter = 1
                dct['butter/oil'] = [x]
            elif 'butter/oil' in dct.keys() and butter ==0 and len(dct['butter/oil'])==1:
                dct['butter/oil'].append(x)
    return pd.Series(dct)

def to_cups(df):
    """
    Takes in dataframe and converts eggs, vanilla, baking powder values to cups
    """
    df = df.copy()
    df['Eggs'] = df['Eggs']/4
    df['Vanilla (tsp)'] = df['Vanilla (tsp)']/48
    df['Baking Powder (tsp)'] = df['Baking Powder (tsp)']/48

    return df.rename(columns={'Eggs': 'Eggs (cups)', 'Vanilla (tsp)': 'Vanilla (cups)',
                              'Baking Powder (tsp)': 'Baking Powder (cups)'})

def clean_data():
    in_file = '././data/recipes.csv'
    out_file = '././data/recipes_clean.csv'
    data_dct = {}
    non_ingredient_columns = ['link','blog','type']

    # read in data, drop nulls, and lowercase strings
    df_org = pd.read_csv(in_file)
    df = df_org.dropna().reset_index(drop=True)
    df['ingredients'] = df['ingredients'].str.lower().str[2:-2].str.split("', '")

    # drop recipes without flour
    has_flour = df['ingredients'].apply(has_ingredient, args=('flour',))
    df = df[has_flour==True].reset_index(drop=True)

    # create one hot dataframe
    df_onehot = pd.concat([df.iloc[:,:2], df['ingredients'].apply(one_hot)], axis=1)
    data_dct['onehot'] = df_onehot

    # create column for blog
    df_onehot = df_onehot.assign(blog=df_onehot['link'].apply(create_blog))

    # clean one hot dataframe
    df_clean = df_onehot[non_ingredient_columns]
    df_clean = df_clean.assign(**{'Flour (cups)':df_onehot['flour'].apply(convert_flour_sugar)})
    df_clean = df_clean.assign(**{'Sugar (cups)':df_onehot['sugar'].apply(convert_flour_sugar)})
    df_clean = df_clean.assign(**{'Eggs':df_onehot['egg'].apply(convert_egg)})
    df_clean = df_clean.assign(**{'Vanilla (tsp)':df_onehot['vanilla'].apply(convert_van)})
    df_clean = df_clean.assign(**{'Baking Powder (tsp)':df_onehot['baking powder/soda'].apply(convert_bpbs)})
    df_clean = df_clean.assign(**{'Butter (cups)':df_onehot['butter/oil'].apply(convert_bo)})
    data_dct['clean'] = df_clean

    # convert eggs, vanilla, baking powder values to cups
    df_cups = to_cups(df_clean)
    data_dct['cups'] = df_cups

    # convert values to percentage amounts
    df_percent = pd.concat([df_clean.iloc[:,:3], df_cups.iloc[:,3:].apply(lambda x:x/x.sum()*100,axis=1)],axis=1)
    df_percent.columns = non_ingredient_columns + [x[:x.find('(')] + '(%)' for x in df_percent.columns if '(' in x]
    data_dct['percent'] = df_percent


    # write to csv
    df_percent.to_csv(out_file, index=False)
    return data_dct
