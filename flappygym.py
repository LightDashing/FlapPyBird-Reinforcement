import gym
from itertools import cycle
import random
import sys
import pygame
import numpy as np
from pygame.locals import *
from stable_baselines3 import PPO


class FlappyEnv(gym.Env):

    def __init__(self):

        self.FPS = 30
        self.SCREENWIDTH = 288
        self.SCREENHEIGHT = 512
        self.mean_crash_y = []

        self.action_space = gym.spaces.Discrete(2, )
        self.max_episode_steps = 40_000
        self.steps = 0

        self.observation_space = gym.spaces.Box(shape=(5,), low=-self.SCREENHEIGHT - 400,
                                                high=self.SCREENHEIGHT + 400, dtype=np.float64)

        self.PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
        self.BASEY = self.SCREENHEIGHT * 0.79
        self.HEIGHT_LIMIT = 50
        # image, sound and hitmask  dicts
        self.IMAGES, self.HITMASKS = {}, {}

        # list of all possible players (tuple of 3 positions of flap)
        self.PLAYERS_LIST = (
            # red bird
            (
                'assets/sprites/redbird-upflap.png',
                'assets/sprites/redbird-midflap.png',
                'assets/sprites/redbird-downflap.png',
            ),
            # blue bird
            (
                'assets/sprites/bluebird-upflap.png',
                'assets/sprites/bluebird-midflap.png',
                'assets/sprites/bluebird-downflap.png',
            ),
            # yellow bird
            (
                'assets/sprites/yellowbird-upflap.png',
                'assets/sprites/yellowbird-midflap.png',
                'assets/sprites/yellowbird-downflap.png',
            ),
        )

        # list of backgrounds
        self.BACKGROUNDS_LIST = (
            'assets/sprites/background-day.png',
            'assets/sprites/background-night.png',
        )

        # list of pipes
        self.PIPES_LIST = (
            'assets/sprites/pipe-green.png',
            'assets/sprites/pipe-red.png',
        )
        pygame.init()
        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
        # pygame.display.quit()
        self.FPSCLOCK = pygame.time.Clock()
        # self.SCREEN = None
        pygame.display.set_caption('Flappy Bird')

        self.IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )

        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

        # sounds
        if 'win' in sys.platform:
            soundExt = '.wav'
        else:
            soundExt = '.ogg'

        randBg = random.randint(0, len(self.BACKGROUNDS_LIST) - 1)
        self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS_LIST[randBg]).convert()

        randPlayer = random.randint(0, len(self.PLAYERS_LIST) - 1)
        self.IMAGES['player'] = (
            pygame.image.load(self.PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        pipeindex = random.randint(0, len(self.PIPES_LIST) - 1)
        self.IMAGES['pipe'] = (
            pygame.transform.flip(
                pygame.image.load(self.PIPES_LIST[pipeindex]).convert_alpha(), False, True),
            pygame.image.load(self.PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hitmask for pipes
        self.HITMASKS['pipe'] = (
            self.get_hitmask(self.IMAGES['pipe'][0]),
            self.get_hitmask(self.IMAGES['pipe'][1]),
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            self.get_hitmask(self.IMAGES['player'][0]),
            self.get_hitmask(self.IMAGES['player'][1]),
            self.get_hitmask(self.IMAGES['player'][2]),
        )

        player_index = 0
        player_index_gen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration
        loopIter = 0

        playerx = int(self.SCREENWIDTH * 0.2)
        playery = int((self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)

        messagex = int((self.SCREENWIDTH - self.IMAGES['message'].get_width()) / 2)
        messagey = int(self.SCREENHEIGHT * 0.12)

        basex = 0
        # amount by which base can maximum shift to left
        self.base_shift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # player shm for up-down motion on welcome screen
        player_shm_vals = {'val': 0, 'dir': 1}

        if (loopIter + 1) % 5 == 0:
            player_index = next(player_index_gen)
        self.loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % self.base_shift)
        self.player_shm(player_shm_vals)

        self.movement_info = {
            'playery': playery + player_shm_vals['val'],
            'basex': basex,
            'playerIndexGen': player_index_gen,
        }

        self.score = self.playerIndex = self.loopIter = 0
        self.player_index_gen = self.movement_info['playerIndexGen']
        self.player_x, self.player_y = int(self.SCREENWIDTH * 0.2), self.movement_info['playery']

        self.basex = self.movement_info['basex']
        self.base_shift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        new_pipe1 = self.get_random_pipe()
        new_pipe2 = self.get_random_pipe()

        # list of upper pipes
        self.upper_pipes = [
            {'x': self.SCREENWIDTH - 15, 'y': new_pipe1[0]['y']},
            {'x': self.SCREENWIDTH - 15 + (self.SCREENWIDTH / 2), 'y': new_pipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lower_pipes = [
            {'x': self.SCREENWIDTH - 15, 'y': new_pipe1[1]['y']},
            {'x': self.SCREENWIDTH - 15 + (self.SCREENWIDTH / 2), 'y': new_pipe2[1]['y']},
        ]

        dt = 0.05
        self.pipeVelX = -128 * dt

        # player velocity, max velocity, downward acceleration, acceleration on flap
        self.playerVelY = -9  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward acceleration
        self.playerRot = 45  # player's rotation
        self.playerVelRot = 3  # angular speed
        self.playerRotThr = 20  # rotation threshold
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False
        self.pipes_crossed = 0

        self.pygame_init = False
        pygame.quit()

    def close(self):
        pygame.quit()
        self.pygame_init = False

    def render(self, mode="human"):
        if mode == "human":
            if not self.pygame_init:
                pygame.init()
                self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
                self.pygame_init = True
            self.SCREEN.blit(self.IMAGES['background'], (0, 0))

            for uPipe, lPipe in zip(self.upper_pipes, self.lower_pipes):
                self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            self.SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))

            visible_rot = self.playerRotThr
            if self.playerRot <= self.playerRotThr:
                visible_rot = self.playerRot

            player_face = pygame.transform.rotate(self.IMAGES['player'][self.playerIndex], visible_rot)
            self.SCREEN.blit(player_face, (self.player_x, self.player_y))

            self.FPSCLOCK.tick(self.FPS)
            pygame.display.update()

    def reset(self):
        self.reset_level()
        self.score = 0
        pipeW = self.IMAGES['pipe'][0].get_width()
        pipeH = self.IMAGES['pipe'][0].get_height()
        bottom_pipe = Rect(self.upper_pipes[0]["x"], self.upper_pipes[0]["y"] + pipeH + 10,
                           pipeW, self.PIPEGAPSIZE - 30)

        second_rect = Rect(self.upper_pipes[1]["x"], self.upper_pipes[1]["y"] + pipeH,
                           pipeW, self.PIPEGAPSIZE)

        obs = np.asarray([self.playerVelY, self.upper_pipes[0]["x"] - self.player_x,
                          bottom_pipe.bottomleft[1] - self.player_y,
                          self.upper_pipes[1]["x"] - self.player_x,
                          second_rect.bottomleft[1] - self.player_y])
        self.steps = 0
        self.pipes_crossed = 0
        return obs

    def reset_level(self):
        player_index = 0
        player_index_gen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration
        self.loopIter = 0

        self.player_x = int(self.SCREENWIDTH * 0.2)
        self.player_y = int((self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)

        message_x = int((self.SCREENWIDTH - self.IMAGES['message'].get_width()) / 2)
        message_y = int(self.SCREENHEIGHT * 0.12)

        basex = 0
        # amount by which base can maximum shift to left
        base_shift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # player shm for up-down motion on welcome screen
        player_shm_vals = {'val': 0, 'dir': 1}

        if (self.loopIter + 1) % 5 == 0:
            player_index = next(player_index_gen)
        self.loopIter = (self.loopIter + 1) % 30
        basex = -((-basex + 4) % base_shift)
        self.player_shm(player_shm_vals)

        self.movement_info = {
            'playery': self.player_y + player_shm_vals['val'],
            'basex': basex,
            'playerIndexGen': player_index_gen,
        }

        self.playerIndex = self.loopIter = 0
        self.player_index_gen = self.movement_info['playerIndexGen']
        self.player_x, self.player_y = int(self.SCREENWIDTH * 0.2), self.movement_info['playery']

        self.basex = self.movement_info['basex']
        self.base_shift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        new_pipe1 = self.get_random_pipe()
        new_pipe2 = self.get_random_pipe()

        # list of upper pipes
        self.upper_pipes = [
            {'x': self.SCREENWIDTH - 15, 'y': new_pipe1[0]['y']},
            {'x': self.SCREENWIDTH - 15 + (self.SCREENWIDTH / 2), 'y': new_pipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lower_pipes = [
            {'x': self.SCREENWIDTH - 15, 'y': new_pipe1[1]['y']},
            {'x': self.SCREENWIDTH - 15 + (self.SCREENWIDTH / 2), 'y': new_pipe2[1]['y']},
        ]

        pipeW = self.IMAGES['pipe'][0].get_width()
        pipeH = self.IMAGES['pipe'][0].get_height()

        pipes = []
        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            pipes.append(pygame.Rect(upper_pipe["x"], upper_pipe["y"], pipeW, pipeH))
            pipes.append(pygame.Rect(lower_pipe["x"], lower_pipe["y"], pipeW, pipeH))

        self.pipes_crossed = 0
        return np.asarray([self.playerVelY])

    def step(self, action):
        self.steps += 1
        if action == 1:
            if self.player_y > -2 * self.IMAGES['player'][0].get_height():
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        crashTest = self.check_crash({'x': self.player_x, 'y': self.player_y, 'index': self.playerIndex},
                                     self.upper_pipes, self.lower_pipes)

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.player_index_gen)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.base_shift)

        # rotate the player
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            self.playerRot = 45

        player_height = self.IMAGES['player'][self.playerIndex].get_height()
        self.player_y += min(self.playerVelY, self.BASEY - self.player_y - player_height)

        # move pipes to left
        for uPipe, lPipe in zip(self.upper_pipes, self.lower_pipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 3 > len(self.upper_pipes) > 0 and 0 < self.upper_pipes[0]['x'] < 5:
            new_pipe = self.get_random_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

        # remove first pipe if its out of the screen
        if len(self.upper_pipes) > 0 and self.upper_pipes[0]['x'] < -self.IMAGES['pipe'][0].get_width():
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        pipeW = self.IMAGES['pipe'][0].get_width()
        pipeH = self.IMAGES['pipe'][0].get_height()

        pipes = []
        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            pipes.append(pygame.Rect(upper_pipe["x"], upper_pipe["y"], pipeW, pipeH))
            pipes.append(pygame.Rect(lower_pipe["x"], lower_pipe["y"], pipeW, pipeH))

        first_rect = Rect(self.upper_pipes[0]["x"], self.upper_pipes[0]["y"] + pipeH,
                          pipeW, self.PIPEGAPSIZE)
        second_rect = Rect(self.upper_pipes[1]["x"], self.upper_pipes[1]["y"] + pipeH,
                           pipeW, self.PIPEGAPSIZE)

        if self.player_x < self.upper_pipes[0]["x"]:
            nearest_pipe_x = self.upper_pipes[0]["x"]
            bottom_rect = first_rect
        else:
            nearest_pipe_x = self.upper_pipes[1]["x"]
            bottom_rect = second_rect

        observation_space = np.asarray([-self.playerVelY, nearest_pipe_x - self.player_x,
                                        bottom_rect.centery - self.player_y,
                                        self.upper_pipes[1]["x"] - self.player_x,
                                        second_rect.centery - self.player_y])
        done = False
        if crashTest["pipe_crash"]:
            done = True
        elif crashTest["ceiling_crash"] or crashTest["floor_crash"]:
            done = True
        else:
            player_mid_pos = self.player_x + self.IMAGES['player'][0].get_width() / 2
            for pipe in self.upper_pipes:
                pipe_mid_pos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
                if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                    self.score += 1 + self.pipes_crossed * 0.1
                    self.pipes_crossed += 1

        if self.steps >= self.max_episode_steps:
            done = True
        return observation_space, self.pipes_crossed, done, {"something": "None"}

    @staticmethod
    def get_hitmask(image):
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask

    @staticmethod
    def player_shm(player_shm):
        """oscillates the value of playerShm['val'] between 8 and -8"""
        if abs(player_shm['val']) == 8:
            player_shm['dir'] *= -1

        if player_shm['dir'] == 1:
            player_shm['val'] += 1
        else:
            player_shm['val'] -= 1

    def get_random_pipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gap_y = random.randrange(0, int(self.BASEY * 0.6 - self.PIPEGAPSIZE))
        gap_y += int(self.BASEY * 0.2)
        pipe_height = self.IMAGES['pipe'][0].get_height()
        pipe_x = self.SCREENWIDTH + 10

        return [
            {'x': pipe_x, 'y': gap_y - pipe_height},  # upper pipe
            {'x': pipe_x, 'y': gap_y + self.PIPEGAPSIZE},  # lower pipe
        ]

    def check_crash(self, player, upper_pipes, lower_pipes):

        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()
        # print(player['w'], player['h'])
        # print(self.player_x, self.player_y)
        # if player['w']

        # if player crashes into ground
        # print(self.BASEY - 1, player['y'] + player['h'])
        if player['y'] + player['h'] >= self.BASEY - 1:
            return {"floor_crash": True, "pipe_crash": False, "ceiling_crash": False}
        elif player['y'] + player['h'] <= -15:
            return {"floor_crash": False, "pipe_crash": False, "ceiling_crash": True}
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                                     player['w'], player['h'])
            pipeW = self.IMAGES['pipe'][0].get_width()
            pipeH = self.IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upper_pipes, lower_pipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixel_collision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixel_collision(playerRect, lPipeRect, pHitMask, lHitmask)

                # print(playerRect, lPipeRect)

                if uCollide or lCollide:
                    return {"floor_crash": False, "pipe_crash": True, "ceiling_crash": False}

        return {"floor_crash": False, "pipe_crash": False, "ceiling_crash": False}

    @staticmethod
    def pixel_collision(rect1, rect2, hit_mask1, hit_mask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hit_mask1[x1 + x][y1 + y] and hit_mask2[x2 + x][y2 + y]:
                    return True
        return False


def evaluate():
    env = FlappyEnv()
    model = PPO.load("model.zip", env)
    vec_env = model.get_env()
    obs = vec_env.reset()
    done_counter = 0
    all_rewards = []
    pipes_count = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        pipes_count += reward
        # Uncomment if you want to render
        # vec_env.render()
        if done:
            done_counter += 1
            all_rewards.append(pipes_count)
            pipes_count = 0
            vec_env.reset()
        if done_counter >= 10:
            vec_env.close()
            break
    print(f"Mean rewards: {np.mean(all_rewards)}")


def main():
    env = FlappyEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    for i in range(10):
        vec_env = model.get_env()
        obs = vec_env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render()
            if done:
                vec_env.close()
                break
        model.learn(total_timesteps=100_000)
        model.save(f"test_model_{i}")


if __name__ == "__main__":
    main()
    evaluate()
