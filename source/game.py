import sys
import time
import random
import math

from memory import MemoryInstance
from point import Point
from actions import StaticActions, MoveAction, WallAction
from model import Model
from agents import TopAgent, BottomAgent
from state import State
from display_game import DisplayGame

import constants
from constants import BoardElement
import pygame

class QuoridorGame:
    """Quoridor displays the game, runs the game actions, keeps track of the game state,
    and allows humans to play the machine."""
    def __init__(self):
        pygame.init()
        static_actions = StaticActions(constants.BOARD_SIZE)
        self.static_actions = static_actions
        self.state = State(static_actions)

        if constants.DISPLAY_GAME:
            self.display_game = DisplayGame()

        print("Setting up agent networks...")
        self.model = Model(self.state.vector_state_size, len(static_actions.all_actions), constants.BATCH_SIZE, constants.RESTORE)
        top_agent = TopAgent(static_actions, self.model)
        bottom_agent = BottomAgent(static_actions, self.model)
        print("completed\n")

        self.agents = {BoardElement.AGENT_BOT: bottom_agent, BoardElement.AGENT_TOP: top_agent}
        self.human_agent = BoardElement.AGENT_TOP

        self.drawing_screen = constants.DISPLAY_GAME
        self.game_delay = constants.INITIAL_GAME_DELAY
        self.only_inference = constants.INITIALLY_USING_ONLY_INFERENCE
        self.human_playing = constants.INITIALLY_HUMAN_PLAYING

        self.sum_game_lengths = 0
        self.games = 0
        self.victories = {BoardElement.AGENT_TOP: 0, BoardElement.AGENT_BOT: 0}
        self.reward_sum = 0
        self.reset()

    def reset(self):
        self.actions_taken = 0
        self.state = State(self.static_actions)
        self.human_action = None
        if self.drawing_screen:
            self.display_game.reset(self.state)

    def run(self):
        self.reset()
        game_over = False
        current_agent = BoardElement.AGENT_BOT if random.random() > 0.5 else BoardElement.AGENT_TOP

        while not game_over:
            if not self.human_playing or (self.human_playing and self.human_action) or current_agent != self.human_agent:
                agent = self.agents[current_agent]
                reward = agent.take_action(self.state, self.only_inference, self.human_action)

                if reward is None:
                    game_over = True
                    break

                self.human_action = None
                self.actions_taken += 1
                self.reward_sum += reward

                current_agent = BoardElement.AGENT_TOP if current_agent == BoardElement.AGENT_BOT else BoardElement.AGENT_BOT

                if self.state.winner:
                    game_over = True
                    self.victories[agent.name] += 1
                    self.games += 1
                    self.sum_game_lengths += self.actions_taken

                if self.drawing_screen:
                    self.display_game.draw_screen(self.state)

            if self.game_delay > 0:
                time.sleep(self.game_delay)

            self.check_pygame_events()

    def check_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    self.drawing_screen = not self.drawing_screen
                    print('drawing screen:', self.drawing_screen)

                elif event.key == pygame.K_f:
                    self.game_delay = 0 if self.game_delay == constants.GAME_DELAY_SECONDS else constants.GAME_DELAY_SECONDS
                    print("game delay:", self.game_delay)

                elif event.key == pygame.K_r:
                    self.only_inference = not self.only_inference
                    print("using inference: ", self.only_inference)

                elif event.key is pygame.K_h:
                    self.human_playing = not self.human_playing
                    print("human playing:", self.human_playing)

            if event.type is pygame.MOUSEBUTTONDOWN and self.drawing_screen and self.human_playing:
                self.human_action = self.get_human_action_index(pygame.mouse.get_pos())

    def print_details(self, games_per_epoch):
        """ Print details on recent statistics to see how training is coming along """
        #self.model.save()

        local_avg_game_length = self.sum_game_lengths / games_per_epoch
        self.sum_game_lengths = 0

        recent_reward_avg = self.reward_sum / games_per_epoch
        self.reward_sum = 0

        print("Top Victories: ", self.victories[BoardElement.AGENT_TOP])
        print("Bot Victories: ", self.victories[BoardElement.AGENT_BOT])
        print("Local Average Game Length: ", local_avg_game_length)
        print("Local Average Game Reward: ", recent_reward_avg)

        print("Local Average Loss: ", self.agents[BoardElement.AGENT_BOT].get_recent_loss())
        print('Exploration Probability:', self.agents[BoardElement.AGENT_TOP].get_exploration_probability())
 