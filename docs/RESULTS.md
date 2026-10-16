# Self-evaluation results, per task

Official evaluator (BEHAVIOR-1K v3.9.3), public test instances 0–9 of all 100 tasks, checkpoint step 135,199.
Score per instance = fraction of the task's goal conditions satisfied at the end (1.0 = full success).
**Task score = mean over its 10 instances, instances not run counted as 0. Overall = mean over the 100 tasks = 0.339.**

Coverage: every task's instance 0 was run; a task with any non-zero score ran all 10 instances; a task stopped after
three consecutive zero-score instances (21 tasks, 147 instances not run). All JSON results and every rollout video:
[VivekChandra/b1k2026-rollout-videos](https://huggingface.co/datasets/VivekChandra/b1k2026-rollout-videos).

Task length class (by the evaluator's time limit): short ≤ 8,000 steps, medium ≤ 15,000, long > 15,000.

| # | task | class | instances run | full successes | best | task score |
|---|---|---|---|---|---|---|
| 1 | `installing_a_modem` | short | 10 | 8 | 1.00 | **0.850** |
| 2 | `putting_dirty_dishes_in_sink` | long | 10 | 7 | 1.00 | **0.850** |
| 3 | `putting_away_toys` | long | 10 | 4 | 1.00 | **0.838** |
| 4 | `collecting_aluminum_cans` | long | 10 | 3 | 1.00 | **0.833** |
| 5 | `dispose_of_glass` | medium | 10 | 6 | 1.00 | **0.800** |
| 6 | `store_batteries` | medium | 10 | 7 | 1.00 | **0.767** |
| 7 | `cook_hot_dogs` | medium | 10 | 7 | 1.00 | **0.750** |
| 8 | `organizing_school_stuff` | long | 10 | 1 | 1.00 | **0.733** |
| 9 | `stacking_wood` | long | 10 | 3 | 1.00 | **0.733** |
| 10 | `turning_on_radio` | short | 10 | 7 | 1.00 | **0.700** |
| 11 | `turning_out_all_lights_before_sleep` | long | 10 | 1 | 1.00 | **0.680** |
| 12 | `laying_tile_floors` | long | 10 | 6 | 1.00 | **0.675** |
| 13 | `moving_boxes_to_storage` | long | 10 | 5 | 1.00 | **0.650** |
| 14 | `tidying_bedroom` | long | 10 | 4 | 1.00 | **0.633** |
| 15 | `store_honey` | medium | 10 | 6 | 1.00 | **0.600** |
| 16 | `setup_a_bar_for_a_cocktail_party` | long | 10 | 0 | 0.90 | **0.590** |
| 17 | `clean_your_rusty_garden_tools` | long | 10 | 4 | 1.00 | **0.580** |
| 18 | `make_rose_centerpieces` | short | 10 | 2 | 1.00 | **0.575** |
| 19 | `re_shelving_library_books` | medium | 10 | 3 | 1.00 | **0.567** |
| 20 | `assembling_gift_baskets` | long | 10 | 0 | 0.75 | **0.562** |
| 21 | `composting_waste` | short | 10 | 4 | 1.00 | **0.550** |
| 22 | `spraying_fruit_trees` | medium | 10 | 4 | 1.00 | **0.550** |
| 23 | `put_together_a_basic_pruning_kit` | long | 10 | 3 | 1.00 | **0.525** |
| 24 | `store_produce` | medium | 10 | 4 | 1.00 | **0.525** |
| 25 | `cook_broccolini` | short | 10 | 1 | 1.00 | **0.518** |
| 26 | `boxing_food_after_dinner` | medium | 10 | 3 | 1.00 | **0.517** |
| 27 | `picking_up_trash` | short | 10 | 1 | 1.00 | **0.500** |
| 28 | `rearrange_your_room` | long | 10 | 4 | 1.00 | **0.500** |
| 29 | `setting_mousetraps` | long | 10 | 0 | 0.67 | **0.500** |
| 30 | `collecting_childrens_toys` | long | 10 | 0 | 0.71 | **0.486** |
| 31 | `sorting_vegetables` | long | 10 | 0 | 0.85 | **0.485** |
| 32 | `dispose_of_batteries` | long | 10 | 2 | 1.00 | **0.475** |
| 33 | `tidying_bathroom` | long | 10 | 2 | 1.00 | **0.475** |
| 34 | `packing_meal_for_delivery` | medium | 10 | 1 | 1.00 | **0.450** |
| 35 | `sweeping_garage` | short | 10 | 4 | 1.00 | **0.450** |
| 36 | `putting_shoes_on_rack` | medium | 10 | 0 | 0.80 | **0.440** |
| 37 | `slicing_vegetables` | long | 10 | 2 | 1.00 | **0.433** |
| 38 | `cook_bacon` | medium | 10 | 1 | 1.00 | **0.429** |
| 39 | `clean_boxing_gloves` | medium | 10 | 4 | 1.00 | **0.400** |
| 40 | `installing_a_fax_machine` | short | 10 | 4 | 1.00 | **0.400** |
| 41 | `make_microwave_popcorn` | short | 10 | 4 | 1.00 | **0.400** |
| 42 | `polishing_shoes` | long | 10 | 0 | 0.67 | **0.400** |
| 43 | `preparing_lunch_box` | medium | 10 | 0 | 0.67 | **0.400** |
| 44 | `scrubbing_bathroom_floor` | short | 10 | 4 | 1.00 | **0.400** |
| 45 | `make_gift_bags_for_baby_showers` | medium | 10 | 0 | 0.67 | **0.383** |
| 46 | `sorting_household_items` | long | 10 | 0 | 0.62 | **0.375** |
| 47 | `cook_a_brisket` | medium | 10 | 3 | 1.00 | **0.367** |
| 48 | `setting_the_fire` | medium | 10 | 2 | 1.00 | **0.362** |
| 49 | `halve_an_egg` | medium | 10 | 2 | 1.00 | **0.360** |
| 50 | `cook_a_frozen_pie` | medium | 10 | 3 | 1.00 | **0.350** |
| 51 | `tidying_living_room` | long | 10 | 0 | 0.75 | **0.350** |
| 52 | `putting_up_Christmas_decorations_inside` | long | 10 | 0 | 0.67 | **0.344** |
| 53 | `organizing_art_supplies` | medium | 10 | 0 | 0.80 | **0.340** |
| 54 | `storing_food` | long | 10 | 0 | 0.62 | **0.338** |
| 55 | `thawing_frozen_food` | medium | 10 | 0 | 0.44 | **0.333** |
| 56 | `carrying_in_groceries` | long | 10 | 0 | 0.50 | **0.325** |
| 57 | `set_up_a_coffee_station_in_your_kitchen` | medium | 10 | 0 | 0.67 | **0.317** |
| 58 | `outfit_a_basic_toolbox` | long | 10 | 2 | 1.00 | **0.314** |
| 59 | `putting_away_Halloween_decorations` | long | 10 | 0 | 0.71 | **0.314** |
| 60 | `chop_an_onion` | medium | 10 | 0 | 0.75 | **0.300** |
| 61 | `putting_dishes_away_after_cleaning` | long | 10 | 0 | 0.50 | **0.293** |
| 62 | `chopping_wood` | long | 10 | 0 | 0.50 | **0.275** |
| 63 | `cleaning_up_plates_and_food` | long | 10 | 1 | 1.00 | **0.243** |
| 64 | `freeze_fruit` | long | 10 | 0 | 0.71 | **0.229** |
| 65 | `rearranging_kitchen_furniture` | medium | 10 | 1 | 1.00 | **0.225** |
| 66 | `setting_the_table` | long | 10 | 0 | 0.75 | **0.225** |
| 67 | `picking_up_toys` | long | 10 | 0 | 0.50 | **0.200** |
| 68 | `unloading_the_car` | long | 10 | 2 | 1.00 | **0.200** |
| 69 | `sorting_bottles_cans_and_paper` | long | 10 | 0 | 0.25 | **0.188** |
| 70 | `cook_brussels_sprouts` | long | 10 | 1 | 1.00 | **0.169** |
| 71 | `bringing_in_wood` | long | 10 | 0 | 0.67 | **0.167** |
| 72 | `hiding_Easter_eggs` | medium | 10 | 1 | 1.00 | **0.167** |
| 73 | `clean_up_your_desk` | long | 10 | 0 | 0.64 | **0.145** |
| 74 | `boxing_books_up_for_storage` | long | 10 | 0 | 0.50 | **0.133** |
| 75 | `wash_dog_toys` | long | 10 | 0 | 0.50 | **0.133** |
| 76 | `sorting_books_on_shelf` | medium | 10 | 0 | 0.27 | **0.091** |
| 77 | `clearing_food_from_table_into_fridge` | long | 10 | 0 | 0.20 | **0.080** |
| 78 | `canning_food` | long | 10 | 0 | 0.10 | **0.060** |
| 79 | `getting_organized_for_work` | long | 10 | 0 | 0.10 | **0.030** |
| 80 | `attach_a_camera_to_a_tripod` | short | 3 | 0 | 0.00 | **0.000** |
| 81 | `bringing_paper_to_recycling` | long | 3 | 0 | 0.00 | **0.000** |
| 82 | `bringing_water` | medium | 3 | 0 | 0.00 | **0.000** |
| 83 | `can_meat` | long | 3 | 0 | 0.00 | **0.000** |
| 84 | `carrying_out_garden_furniture` | long | 3 | 0 | 0.00 | **0.000** |
| 85 | `clean_a_keyboard` | short | 3 | 0 | 0.00 | **0.000** |
| 86 | `clean_a_patio` | long | 3 | 0 | 0.00 | **0.000** |
| 87 | `clean_a_trumpet` | short | 3 | 0 | 0.00 | **0.000** |
| 88 | `clean_up_broken_glass` | medium | 3 | 0 | 0.00 | **0.000** |
| 89 | `cleaning_up_branches_and_twigs` | long | 3 | 0 | 0.00 | **0.000** |
| 90 | `cook_cabbage` | long | 3 | 0 | 0.00 | **0.000** |
| 91 | `freeze_pies` | long | 3 | 0 | 0.00 | **0.000** |
| 92 | `hanging_pictures` | short | 3 | 0 | 0.00 | **0.000** |
| 93 | `installing_a_scanner` | short | 3 | 0 | 0.00 | **0.000** |
| 94 | `installing_smoke_detectors` | short | 3 | 0 | 0.00 | **0.000** |
| 95 | `loading_the_car` | long | 3 | 0 | 0.00 | **0.000** |
| 96 | `make_cabinet_doors` | short | 3 | 0 | 0.00 | **0.000** |
| 97 | `make_pizza` | long | 3 | 0 | 0.00 | **0.000** |
| 98 | `spraying_for_bugs` | medium | 3 | 0 | 0.00 | **0.000** |
| 99 | `vacuuming_floors` | short | 3 | 0 | 0.00 | **0.000** |
| 100 | `wash_a_baseball_cap` | medium | 3 | 0 | 0.00 | **0.000** |
