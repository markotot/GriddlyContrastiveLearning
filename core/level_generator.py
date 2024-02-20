import numpy as np
from griddly.util.rllib.environment.level_generator import LevelGenerator

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

def generate_levels(num_levels, seed, width, height):
    config = {
        'width': width,
        'height': height,
        'seed': seed,
    }

    level_generator = ClustersLevelGenerator(config)
    level_strings = []
    for n in range(num_levels):
        level_strings.append(level_generator.generate())
    return level_strings
