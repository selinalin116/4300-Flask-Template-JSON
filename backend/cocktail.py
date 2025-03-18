import requests
import json

ingredient_drink_index = {}

ingredient_url = "https://www.thecocktaildb.com/api/json/v1/1/list.php?i=list"
ingredient_response = requests.get(ingredient_url)
ingredients = ingredient_response.json().get("drinks", [])

# format of the ingredient_url website
# "drinks":[{"strIngredient1":"Light rum"},{"strIngredient1":"Bourbon"},
# {"strIngredient1":"Vodka"},{"strIngredient1":"Gin"},{"strIngredient1":"Blended whiskey"},{"strIngredient1":"Tequila"}
for x in ingredients:
    ingredient = x["strIngredient1"]
    drinks_url = f"https://www.thecocktaildb.com/api/json/v1/1/filter.php?i={ingredient}"
    # here's what we get w/ ingredient = vodka
    # "drinks":[{"strDrink":"155 Belmont","strDrinkThumb":"https:\/\/www.thecocktaildb.com\/images\/media\/drink\/yqvvqs1475667388.jpg",
    # "idDrink":"15346"},{"strDrink":"501 Blue","strDrinkThumb":"https:\/\/www.thecocktaildb.com\/images\/media\/drink\/ywxwqs1461867097.jpg",
    # "idDrink":"17105"},{"strDrink":"57 Chevy with a White License 

    drinks_response = requests.get(drinks_url)
    # get drink names
    drink_list = []
    for y in drinks_response.json().get("drinks", []):
        drink = y["strDrink"]
        drink_list.append(drink)
    ingredient_drink_index[ingredient] = drink_list

# write to file
with open("ingredient_drink_index.json", "w") as f:
    json.dump(ingredient_drink_index, f, indent=4)