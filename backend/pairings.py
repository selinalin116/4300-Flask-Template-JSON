import json
import numpy as np

with open("data/taste_trios.json", "r") as f:
    compatible_trios = json.load(f)

def cosine_sim(word1, word2, model):
    if word1 in model and word2 in model:
        v1, v2 = model[word1], model[word2]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 0   


# def get_pairing_score_ranked(cocktail_ings, recipe_ings, model, sim_threshold=0.30):
#     cocktail_ings = set(ing.lower().strip() for ing in cocktail_ings)
#     recipe_ings = set(ing.lower().strip() for ing in recipe_ings)
#     all_ings = cocktail_ings | recipe_ings

#     best_label = "No Notable Pairing"
#     best_rank = 0

#     for key, compatibility in compatible_trios.items():
#         trio_ings = set(key.split(", "))

#         overlap = set()
#         for ing in all_ings:
#             for trio_ing in trio_ings:
#                 # Exact or substring match
#                 if ing in trio_ing or trio_ing in ing:
#                     overlap.add(trio_ing)
#                 else:
#                     # Semantic similarity check
#                     sim = cosine_sim(ing, trio_ing, model)
#                     if sim >= sim_threshold:
#                         overlap.add(trio_ing)

#         print(overlap)
#         # print(overlap & cocktail_ings)
#         if overlap & cocktail_ings and overlap & recipe_ings:
#             print("here")
#             if len(overlap) >= 3:
#                 return compatibility, 3 if compatibility == "Highly Compatible" else 2
#             elif len(overlap) == 2:
#                 best_label = "Weak Compatibility"
#                 best_rank = 1

#     return best_label, best_rank

def get_pairing_score_ranked(cocktail_ings, recipe_ings, model, sim_threshold=0.30):
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
                elif cosine_sim(ing, trio_ing, model) >= sim_threshold:
                    matched_cocktail_ings.add(trio_ing)
                    break

        for ing in recipe_ings:
            for trio_ing in trio_ings:
                if ing in trio_ing or trio_ing in ing:
                    matched_recipe_ings.add(trio_ing)
                    break
                elif cosine_sim(ing, trio_ing, model) >= sim_threshold:
                    matched_recipe_ings.add(trio_ing)
                    break

        overlap = matched_cocktail_ings | matched_recipe_ings

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