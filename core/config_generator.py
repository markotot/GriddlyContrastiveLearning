import copy

import yaml

config_backgrounds = {
    "floor": "oryx/oryx_fantasy/floor1-2.png",
    "grass": "gvgai/oryx/grass_15.png",
    "orange": "gvgai/oryx/backOrange.png",
    "brown": "gvgai/oryx/backLBrown.png",
    "blue": "gvgai/oryx/backLBlue.png",
    "biege": "gvgai/oryx/backBiege.png",
    "space": "gvgai/oryx/space5.png",
    "grey": "gvgai/oryx/backGrey.png",
    "red": "gvgai/oryx/backRed.png",
    "fill": "block_shapes/fill.png"
}

config_avatars = {
    "knight": "gvgai/oryx/knight1.png",
    "alien": "gvgai/oryx/alien1.png",
    "angel": "gvgai/oryx/angel1.png",
    "burger": "gvgai/newset/burger.png",
    "chef": "gvgai/newset/chef_1.png",
    "coins": "gvgai/oryx/coins1.png",
    "fireman": "gvgai/newset/fireman.png",
    "necromancer": "gvgai/oryx/necromancer1.png",
    "rogue": "gvgai/oryx/rogue_1.png",
    "wolf": "gvgai/oryx/wolf1.png",
}

config_walls = {
    "wall": "oryx/oryx_fantasy/wall1-0.png",
    "barrel": "gvgai/oryx/barrel1.png",
    "door": "gvgai/oryx/door2.png",
    "tree": "gvgai/oryx/tree2.png",
    "evil_tree": "gvgai/oryx/evilTree1.png",
    "fence": "sprite2d/fence.png",
    "fire": "gvgai/oryx/fire1.png",
    "mineral": "gvgai/oryx/mineral1.png",
    "number": "gvgai/newset/c3.png",
    "pipe": "gvgai/newset/pipeUD.png",
}

config_boxes = {
    "armour": {
        "red_box": "oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_armor_chest_red.png",
        "red_block": "oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_armor_boots_red.png",
        "green_box": "oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_armor_gloves_grey.png",
        "green_block": "oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_armor_chest_grey.png",
        "blue_box": "oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_armor_chest_blue.png",
        "blue_block": "oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_armor_boots_blue.png",
    },
    "boxes": {
        "red_box": "gvgai/newset/blockR.png",
        "red_block": "gvgai/newset/blockR2.png",
        "green_box": "gvgai/newset/blockG.png",
        "green_block": "gvgai/newset/blockG2.png",
        "blue_box": "gvgai/newset/blockB.png",
        "blue_block": "gvgai/newset/blockB2.png",
    },
    "boxes2": {
        "red_box": "gvgai/newset/blockT.png",
        "red_block": "gvgai/newset/blockT2.png",
        "green_box": "gvgai/newset/blockG.png",
        "green_block": "gvgai/newset/blockG2.png",
        "blue_box": "gvgai/newset/blockY.png",
        "blue_block": "gvgai/newset/blockY2.png",
    },
    "cars": {
        "red_box": "gvgai/newset/car1.png",
        "red_block": "gvgai/oryx/potion3.png",
        "green_box": "gvgai/newset/blockG.png",
        "green_block": "gvgai/newset/blockG2.png",
        "blue_box": "gvgai/newset/car4.png",
        "blue_block": "gvgai/oryx/potion1.png",
    },
    "chess": {
        "red_box": "gvgai/newset/pawn1R.png",
        "red_block": "gvgai/newset/queenR.png",
        "green_box": "gvgai/newset/blockG.png",
        "green_block": "gvgai/newset/blockG2.png",
        "blue_box": "gvgai/newset/pawn1B.png",
        "blue_block": "gvgai/newset/queenB.png",
    },
    "food": {
        "red_box": "gvgai/oryx/slime3.png",
        "red_block": "gvgai/oryx/mushroom2.png",
        "green_box": "gvgai/newset/blockG.png",
        "green_block": "gvgai/newset/blockG2.png",
        "blue_box": "gvgai/oryx/slime1.png",
        "blue_block": "gvgai/oryx/mushroom1.png",
    },
}

config_levels = {
    0:
        "\
        w w w w w w w\n\
        w . . . . . w\n\
        w 1 1 2 . 2 w\n\
        w . . . A . w\n\
        w a 1 . . 2 w\n\
        w . . h . b w\n\
        w w w w w w w",
    1:
        "\
        w w w w w w w\n\
        w . . . . . w\n\
        w . 1 h 2 h w\n\
        w . . . A . w\n\
        w a 1 . 2 . w\n\
        w . . . . b w\n\
        w w w w w w w",
}


def get_background_env_configs(template_name, train):
    experiment_type = "generated"
    env_configs = [
        f"{experiment_type}/{template_name}-floor-wall-boxes-knight.yaml",
        f"{experiment_type}/{template_name}-biege-wall-boxes-knight.yaml",
        f"{experiment_type}/{template_name}-blue-wall-boxes-knight.yaml",
        f"{experiment_type}/{template_name}-brown-wall-boxes-knight.yaml",
        f"{experiment_type}/{template_name}-fill-wall-boxes-knight.yaml",
        f"{experiment_type}/{template_name}-grass-wall-boxes-knight.yaml",
        f"{experiment_type}/{template_name}-grey-wall-boxes-knight.yaml",
        f"{experiment_type}/{template_name}-orange-wall-boxes-knight.yaml",
        f"{experiment_type}/{template_name}-space-wall-boxes-knight.yaml",
        f"{experiment_type}/{template_name}-red-wall-boxes-knight.yaml",
    ]
    if train:
        env_configs = env_configs[:9]

    return env_configs, env_configs[-1]


def get_mixed_env_configs(template_name, train):
    experiment_type = "generated"
    env_configs = [
        f"{experiment_type}/{template_name}-floor-wall-boxes-knight.yaml",
        f"{experiment_type}/{template_name}-grass-barrel-armour-alien.yaml",
        f"{experiment_type}/{template_name}-orange-door-boxes2-angel.yaml",
        f"{experiment_type}/{template_name}-brown-tree-cars-burger.yaml",
        f"{experiment_type}/{template_name}-blue-evil_tree-chess-chef.yaml",
        f"{experiment_type}/{template_name}-red-fence-food-coins.yaml",
    ]
    if train:
        env_configs = env_configs[:5]
    return env_configs, env_configs[-1]


def set_levels(env_config, levels):
    env_config["Environment"]["Levels"] = levels


def set_background_colour(env_config, background_path="gvgai/oryx/floor1.png"):
    env_config["Environment"]["Observers"]["Sprite2D"]["BackgroundTile"] = background_path


def set_player_avatar(env_config, avatar_path="gvgai/oryx/knight1/png"):
    objects = env_config["Objects"]
    for idx, object in enumerate(objects):
        if object["Name"] == "avatar":
            object["Observers"]["Sprite2D"][0]["Image"] = avatar_path
            env_config["Objects"][idx] = object
            break


def set_walls(env_config, background_path="oryx/oryx_fantasy/wall1.png"):

    objects = env_config["Objects"]
    for idx, object in enumerate(objects):
        if object["Name"] == "wall":
            images = object["Observers"]["Sprite2D"][0]["Image"]
            env_config["Objects"][idx]["Observers"]["Sprite2D"][0]["Image"] = [background_path for _ in images]
            break


def set_boxes(env_config, boxes_paths):
    objects = env_config["Objects"]
    for idx, object in enumerate(objects):

        if object["Name"] == "red_box":
            object["Observers"]["Sprite2D"][0]["Image"] = boxes_paths["red_box"]
            env_config["Objects"][idx] = object
        if object["Name"] == "red_block":
            object["Observers"]["Sprite2D"][0]["Image"] = boxes_paths["red_block"]
            env_config["Objects"][idx] = object

        if object["Name"] == "green_box":
            object["Observers"]["Sprite2D"][0]["Image"] = boxes_paths["green_box"]
            env_config["Objects"][idx] = object
        if object["Name"] == "green_block":
            object["Observers"]["Sprite2D"][0]["Image"] = boxes_paths["green_block"]
            env_config["Objects"][idx] = object

        if object["Name"] == "blue_box":
            object["Observers"]["Sprite2D"][0]["Image"] = boxes_paths["blue_box"]
            env_config["Objects"][idx] = object
        if object["Name"] == "blue_block":
            object["Observers"]["Sprite2D"][0]["Image"] = boxes_paths["blue_block"]
            env_config["Objects"][idx] = object


def different_backgrounds():

    backgrounds = ["floor", "grass", "orange", "brown", "blue", "biege", "space", "grey", "red", "fill"]
    avatars = ["knight"]
    walls = ["wall"]
    boxes = ["boxes"]

    return backgrounds, avatars, walls, boxes


def all_combinations():
    backgrounds = ["floor", "grass", "orange", "brown", "blue", "biege", "space", "grey", "red", "fill"]
    walls = ["wall", "barrel", "door", "tree", "evil_tree", "fence", "fire", "mineral", "number", "pipe"]
    boxes = ["armour", "boxes", "boxes2", "cars", "chess", "food"]
    avatars = ["knight", "alien", "angel", "burger", "chef", "coins", "fireman", "necromancer", "rogue", "wolf"]

    return backgrounds, avatars, walls, boxes


def generate_configs(template_name, backgrounds, avatars, walls, boxes, levels):
    with open(f"../configs/{template_name}.yaml", "r") as file:
        template = yaml.safe_load(file)

    for background in backgrounds:
        for avatar in avatars:
            for wall in walls:
                for box in boxes:
                    env_config = copy.deepcopy(template)

                    # set_levels(env_config, [config_levels[x] for x in levels])
                    set_background_colour(env_config, config_backgrounds[background])
                    set_player_avatar(env_config, config_avatars[avatar])
                    set_walls(env_config, config_walls[wall])
                    set_boxes(env_config, config_boxes[box])

                    name = f"{template_name}-{background}-{wall}-{box}-{avatar}.yaml"
                    with open(f"../configs/generated/{name}", "w") as f:
                        yaml.dump(env_config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":


    template_name = "dwarf"
    levels = [0, 1]
    backgrounds, avatars, walls, boxes = different_backgrounds()
    # backgrounds, avatars, walls, boxes = all_combinations()
    generate_configs(template_name, backgrounds, avatars, walls, boxes, levels)

