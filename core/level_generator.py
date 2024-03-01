import numpy as np
from griddly.util.rllib.environment.level_generator import LevelGenerator


class DwarfLevelGenerator(LevelGenerator):

    GOAL = 'g'
    DOOR = 'D'
    KEY = 'k'
    AVATAR = 'A'
    WALL = 'W'

    def __init__(self, config):
        super().__init__(config)


        self._width = config.get('width', 10)
        self._height = config.get('height', 10)

        self.seed = config.get('seed', 0)

        np.random.seed(self.seed)

    def _place_walls(self, map, vertical):

        # top/bottom wall
        wall_y = np.array([0, self._height - 1])
        map[:, wall_y] = DwarfLevelGenerator.WALL

        # left/right wall
        wall_x = np.array([0, self._width - 1])
        map[wall_x, :] = DwarfLevelGenerator.WALL

        #middle wall

        if vertical:
            split_point = np.random.randint(low=2, high=self._height - 2)
            for x in range(1, self._width - 1):
                map[x, split_point] = DwarfLevelGenerator.WALL

            door_position = np.random.randint(low=1, high=self._height - 1)
            map[door_position, split_point] = DwarfLevelGenerator.DOOR
        else:
            split_point = np.random.randint(low=2, high=self._width - 2)
            for y in range(1, self._height - 1):
                map[split_point, y] = DwarfLevelGenerator.WALL

            door_position = np.random.randint(low=1, high=self._height - 1)
            map[split_point, door_position] = DwarfLevelGenerator.DOOR

        return map, split_point

    def place_objects(self, map, possible_locations, split_point, vertical):


        if vertical:
            left = np.random.choice([True, False], size=1)
            if left:
                key_pos_x = np.random.randint(low=1, high=split_point)
                key_pos_y = np.random.randint(low=1, high=self._height - 1)

                while True:
                    agent_pos_x = np.random.randint(low=1, high=split_point)
                    agent_pos_y = np.random.randint(low=1, high=self._height - 1)
                    if agent_pos_x != key_pos_x or agent_pos_y != key_pos_y:
                        break

                goal_pos_x = np.random.randint(low=split_point + 1, high=self._width - 1)
                goal_pos_y = np.random.randint(low=1, high=self._height - 1)
            else:
                key_pos_x = np.random.randint(low=split_point + 1, high=self._height - 1)
                key_pos_y = np.random.randint(low=1, high=self._height - 1)

                while True:
                    agent_pos_x = np.random.randint(low=split_point + 1, high=self._height - 1)
                    agent_pos_y = np.random.randint(low=1, high=self._height - 1)
                    if agent_pos_x != key_pos_x or agent_pos_y != key_pos_y:
                        break

                goal_pos_x = np.random.randint(low=1, high=split_point)
                goal_pos_y = np.random.randint(low=1, high=self._height - 1)

        else:
            up = np.random.choice([True, False], size=1)
            if up:
                key_pos_x = np.random.randint(low=1, high=self._width - 1)
                key_pos_y = np.random.randint(low=1, high=split_point)

                del possible_locations[possible_locations.index([key_pos_x, key_pos_y])]

                while True:
                    agent_pos_x = np.random.randint(low=1, high=self._width - 1)
                    agent_pos_y = np.random.randint(low=1, high=split_point)
                    if agent_pos_x != key_pos_x or agent_pos_y != key_pos_y:
                        break

                goal_pos_x = np.random.randint(low=1, high=self._width - 1)
                goal_pos_y = np.random.randint(low=split_point + 1, high=self._height - 1)

            else:
                key_pos_x = np.random.randint(low=1, high=self._width - 1)
                key_pos_y = np.random.randint(low=split_point + 1, high=self._height - 1)

                while True:
                    agent_pos_x = np.random.randint(low=1, high=self._width - 1)
                    agent_pos_y = np.random.randint(low=split_point + 1, high=self._height - 1)
                    if agent_pos_x != key_pos_x or agent_pos_y != key_pos_y:
                        break

                goal_pos_x = np.random.randint(low=1, high=self._width - 1)
                goal_pos_y = np.random.randint(low=1, high=split_point)

        map[key_pos_y, key_pos_x] = DwarfLevelGenerator.KEY
        map[goal_pos_y, goal_pos_x] = DwarfLevelGenerator.GOAL
        map[agent_pos_y, agent_pos_x] = DwarfLevelGenerator.AVATAR

    def generate(self):
        map = np.chararray((self._width, self._height), itemsize=2)
        map[:] = '.'
        vertical = np.random.choice([True, False], size=1)
        # Generate walls
        map, split_point = self._place_walls(map, vertical)

        # all possible locations
        possible_locations = []
        for w in range(1, self._width - 1):
            for h in range(1, self._height - 1):
                if map[w, h] != DwarfLevelGenerator.WALL:
                    possible_locations.append([w, h])



        self.place_objects(map, possible_locations, split_point, vertical)

        level_string = ''
        for h in range(0, self._height):
            for w in range(0, self._width):
                level_string += map[w, h].decode().ljust(4)
            level_string += '\n'

        return level_string
class ButterfliesLevelGenerator(LevelGenerator):

    SPIDER = 'S'
    BUTTERFLY = '1'
    COCOON = '0'
    CATCHER = 'A'
    WALL = 'w'

    def __init__(self, config):
        super().__init__(config)


        self._width = config.get('width', 10)
        self._height = config.get('height', 10)
        self.min_spiders = config.get('min_num_spiders', 0)
        self.max_spiders = config.get('max_num_spiders', 2)
        self.min_butterflies = config.get('min_num_butterflies', 2)
        self.max_butterflies = config.get('max_num_butterflies', 5)
        self.min_cocoons = config.get('min_num_cocoons', 0)
        self.max_cocoons = config.get('max_num_cocoons', 2)
        self.seed = config.get('seed', 0)

        self.min_num_walls = config.get('min_num_walls', 0)
        self.max_num_walls = config.get('max_num_walls', 5)

        np.random.seed(self.seed)
    def _place_walls(self, map):

        # top/bottom wall
        wall_y = np.array([0, self._height - 1])
        map[:, wall_y] = ButterfliesLevelGenerator.WALL

        # left/right wall
        wall_x = np.array([0, self._width - 1])
        map[wall_x, :] = ButterfliesLevelGenerator.WALL

        num_walls = np.random.randint(low=self.min_num_walls, high=self.max_num_walls)

        x = np.random.randint(low=1, high=self._width-1, size=num_walls)
        y = np.random.randint(low=1, high=self._height-1, size=num_walls)

        for x, y in zip(x, y):
            map[x, y] = ButterfliesLevelGenerator.WALL

        return map

    def place_spiders(self, map, possible_locations, spider_char):

        num_spiders = np.random.randint(low=self.min_spiders, high=self.max_spiders)
        for idx in range(num_spiders):
            spider_location_idx = np.random.choice(len(possible_locations))
            spider_location = possible_locations[spider_location_idx]
            del possible_locations[spider_location_idx]
            map[spider_location[0], spider_location[1]] = spider_char

        return map, possible_locations

    def place_cocoon(self, map, possible_locations, cocoon_char):

        num_cocoon = np.random.randint(low=self.min_cocoons, high=self.max_cocoons)
        for idx in range(num_cocoon):
            cocoon_location_idx = np.random.choice(len(possible_locations))
            cocoon_location = possible_locations[cocoon_location_idx]
            del possible_locations[cocoon_location_idx]
            map[cocoon_location[0], cocoon_location[1]] = cocoon_char

        return map, possible_locations

    def place_butterflies(self, map, possible_locations, butterfly_char):

        num_butterfly = np.random.randint(low=self.min_butterflies, high=self.max_butterflies)
        for idx in range(num_butterfly):
            butterfly_location_idx = np.random.choice(len(possible_locations))
            butterfly_location = possible_locations[butterfly_location_idx]
            del possible_locations[butterfly_location_idx]
            map[butterfly_location[0], butterfly_location[1]] = butterfly_char

        return map, possible_locations

    def generate(self):
        map = np.chararray((self._width, self._height), itemsize=2)
        map[:] = '.'

        # Generate walls
        map = self._place_walls(map)

        # all possible locations
        possible_locations = []
        for w in range(1, self._width - 1):
            for h in range(1, self._height - 1):
                if map[w, h] != ButterfliesLevelGenerator.WALL:
                    possible_locations.append([w, h])

        # Place spiders
        map, possible_locations = self.place_spiders(
            map,
            possible_locations,
            ButterfliesLevelGenerator.SPIDER,
        )

        map, possible_locations = self.place_cocoon(
            map,
            possible_locations,
            ButterfliesLevelGenerator.COCOON,
        )

        map, possible_locations = self.place_butterflies(
            map,
            possible_locations,
            ButterfliesLevelGenerator.BUTTERFLY,
        )

        # Place Agent
        agent_location_idx = np.random.choice(len(possible_locations))
        agent_location = possible_locations[agent_location_idx]
        map[agent_location[0], agent_location[1]] = ButterfliesLevelGenerator.CATCHER

        level_string = ''
        for h in range(0, self._height):
            for w in range(0, self._width):
                level_string += map[w, h].decode().ljust(4)
            level_string += '\n'

        return level_string


class ClustersLevelGenerator(LevelGenerator):
    BLUE_BLOCK = 'a'
    BLUE_BOX = '1'
    RED_BLOCK = 'b'
    RED_BOX = '2'
    GREEN_BLOCK = 'c'
    GREEN_BOX = '3'

    AGENT = 'A'

    WALL = 'w'
    SPIKES = 'h'

    def __init__(self, config):
        super().__init__(config)
        self._width = config.get('width', 10)
        self._height = config.get('height', 10)
        self._p_red = config.get('p_red', 1.0)
        self._p_green = config.get('p_green', 0)
        self._p_blue = config.get('p_blue', 1.0)
        self._m_red = config.get('m_red', 8)
        self._m_blue = config.get('m_blue', 8)
        self._m_green = config.get('m_green', 1)
        self._m_spike = config.get('m_spike', 1)
        self.seed = config.get('seed', 0)
        np.random.seed(self.seed)
    def _place_walls(self, map):

        # top/bottom wall
        wall_y = np.array([0, self._height - 1])
        map[:, wall_y] = ClustersLevelGenerator.WALL

        # left/right wall
        wall_x = np.array([0, self._width - 1])
        map[wall_x, :] = ClustersLevelGenerator.WALL

        return map

    def _place_blocks_and_boxes(self, map, possible_locations, p, block_char, box_char, max_boxes):
        if np.random.random() < p:
            block_location_idx = np.random.choice(len(possible_locations))
            block_location = possible_locations[block_location_idx]
            del possible_locations[block_location_idx]
            map[block_location[0], block_location[1]] = block_char

            num_boxes = 1 + np.random.choice(max_boxes - 1)
            for k in range(num_boxes):
                box_location_idx = np.random.choice(len(possible_locations))
                box_location = possible_locations[box_location_idx]
                del possible_locations[box_location_idx]
                map[box_location[0], box_location[1]] = box_char

        return map, possible_locations

    def generate(self):
        map = np.chararray((self._width, self._height), itemsize=2)
        map[:] = '.'

        # Generate walls
        map = self._place_walls(map)

        # all possible locations
        possible_locations = []
        for w in range(1, self._width - 1):
            for h in range(1, self._height - 1):
                possible_locations.append([w, h])

        # Place Red
        map, possible_locations = self._place_blocks_and_boxes(
            map,
            possible_locations,
            self._p_red,
            ClustersLevelGenerator.RED_BLOCK,
            ClustersLevelGenerator.RED_BOX,
            self._m_red
        )

        # Place Blue
        map, possible_locations = self._place_blocks_and_boxes(
            map,
            possible_locations,
            self._p_blue,
            ClustersLevelGenerator.BLUE_BLOCK,
            ClustersLevelGenerator.BLUE_BOX,
            self._m_blue
        )

        # # Place Green
        # map, possible_locations = self._place_blocks_and_boxes(
        #     map,
        #     possible_locations,
        #     self._p_green,
        #     ClustersLevelGenerator.GREEN_BLOCK,
        #     ClustersLevelGenerator.GREEN_BOX,
        #     self._m_green
        # )

        # Place Spikes
        num_spikes = np.random.choice(self._m_spike)
        for k in range(num_spikes):
            spike_location_idx = np.random.choice(len(possible_locations))
            spike_location = possible_locations[spike_location_idx]
            del possible_locations[spike_location_idx]
            map[spike_location[0], spike_location[1]] = ClustersLevelGenerator.SPIKES

        # Place Agent
        agent_location_idx = np.random.choice(len(possible_locations))
        agent_location = possible_locations[agent_location_idx]
        map[agent_location[0], agent_location[1]] = ClustersLevelGenerator.AGENT

        level_string = ''
        for h in range(0, self._height):
            for w in range(0, self._width):
                level_string += map[w, h].decode().ljust(4)
            level_string += '\n'

        return level_string

def generate_levels(game_name, num_levels, seed, width, height):
    config = {
        'width': width,
        'height': height,
        'seed': seed,
    }

    if game_name == "butterflies":
        level_generator = ButterfliesLevelGenerator(config)
    elif game_name == "cluster":
        level_generator = ClustersLevelGenerator(config)
    elif game_name == "dwarf":
        level_generator = DwarfLevelGenerator(config)

    level_strings = []
    for n in range(num_levels):
        level_strings.append(level_generator.generate())
    return level_strings

level_strings = generate_levels("dwarf", 5, 0, 7, 7)