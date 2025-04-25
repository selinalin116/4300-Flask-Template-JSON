import json
from helper_functions import cosine_sim

with open("data/taste_trios.json", "r") as f:
    compatible_trios = json.load(f)

def get_pairing_score_ranked(cocktail_ings, recipe_ings, model, sim_threshold=0.70):
    cocktail_ings = set(ing.lower().strip() for ing in cocktail_ings)
    recipe_ings = set(ing.lower().strip() for ing in recipe_ings)

    best_label = "No Notable Pairing"
    best_rank = 0

    for key, compatibility in compatible_trios.items():
        trio_ings = set(key.split(", "))

        matched_cocktail_ings = set()
        matched_recipe_ings = set()

        for ing in cocktail_ings:
            for trio_ing in trio_ings:
                if ing in trio_ing or trio_ing in ing:
                    matched_cocktail_ings.add(trio_ing)
                    break
                else:
                    if ing in model and trio_ing in model:
                        v1, v2 = model[ing], model[trio_ing]
                        sim = cosine_sim(v1, v2)
                        if sim >= sim_threshold:
                            matched_cocktail_ings.add(trio_ing)
                            break

        for ing in recipe_ings:
            for trio_ing in trio_ings:
                if ing in trio_ing or trio_ing in ing:
                    matched_recipe_ings.add(trio_ing)
                    break
                else:
                    if ing in model and trio_ing in model:
                        v1, v2 = model[ing], model[trio_ing]
                        sim = cosine_sim(v1, v2)
                        if sim >= sim_threshold:
                            matched_recipe_ings.add(trio_ing)
                            break

        overlap = matched_cocktail_ings | matched_recipe_ings
        ingredients_overlap = overlap

        if matched_cocktail_ings and matched_recipe_ings:
            if len(overlap) >= 3:
                if compatibility == "Highly Compatible":
                    return compatibility, 3
                elif compatibility == "Moderately Compatible":
                    return compatibility, 2.5
                elif compatibility == "Compatible":
                    return compatibility, 2
            elif len(overlap) == 2:
                best_label = "Weak Compatibility"
                best_rank = 1


    return best_label, best_rank