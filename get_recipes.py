import pprint
import re
import json
import itertools
from collections import Counter
from train_model import class_names

# # clean the json file from empty entries
# with open("recipes/recipes_raw_nosource_ar.json", "r") as f:
#     json_data = json.load(f)

# empty_keys = []
# for k, v in json_data.items():
#     if not v:
#         empty_keys.append(k)

# for key in empty_keys:
#     del json_data[key]

# with open("recipes/recipes_raw_nosource_ar.json", "w") as f:
#     json.dump(json_data, f, indent=4)

f = open("recipes_raw_nosource_ar.json")
json_data = json.load(f)

grouped_recipes = {}

for recipe_id, recipe_details in json_data.items():
    title = recipe_details["title"]

    matching_food_items = [
        " ".join(food_name.split("_"))
        for food_name in class_names
        if " ".join(food_name.split("_")).lower() in title.lower()
    ]

    for food_name in matching_food_items:
        if food_name in grouped_recipes:
            grouped_recipes[food_name].append(recipe_details)
        else:
            grouped_recipes[food_name] = [
                recipe_details
            ]  # list so that we can append to it later on

# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint([grouped_recipes["apple pie"][0]["title"]])


# for i in json_data:
#     for name in class_names:
#         full_name = " ".join(name.split("_"))
#         if full_name in json_data[i]["title"].lower():
#             recipes[full_name] = json_data[i]


def get_food_titles(food_name):
    """
    Returns a list of different recipes for the food_name.
    """
    return set(
        [
            grouped_recipes[food_name][i]["title"]
            for i in range(len(grouped_recipes[food_name]))
        ]
    )


def get_recipe(food_name, food_recipe_name):
    """
    Gets a recipe of a given dish.

    Returns:
        ingredients (what the dish consists of): str
        instructions (how to cook the dish): str
    """
    full_name = " ".join(food_name.split("_"))
    if full_name in grouped_recipes:
        ingredients = [
            grouped_recipes[food_name][i]["ingredients"]
            for i in range(len(grouped_recipes[food_name]))
            if grouped_recipes[food_name][i]["title"] == food_recipe_name
        ][0]
        ingredients = ",\n".join(ingredients).title()

        instructions = [
            grouped_recipes[food_name][i]["instructions"]
            for i in range(len(grouped_recipes[food_name]))
            if grouped_recipes[food_name][i]["title"] == food_recipe_name
        ][0].strip()

        return ingredients, instructions


if __name__ == "__main__":
    pass  # get_recipe("steak")
