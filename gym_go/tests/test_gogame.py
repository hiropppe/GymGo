import pytest
import numpy as np

from gym_go import gogame
from gym_go import govars

np.set_printoptions(threshold=np.inf)


@pytest.fixture(scope='function', autouse=True)
def init_state():
    state = gogame.init_state(9) 
    yield state


def test_shift_board_history(init_state):
    state = init_state
    moves = [[], []]
    x, y = govars.BLACK, govars.WHITE

    def step(state):
        action2d, state = random_action2d(state)
        moves[x].append(action2d)
        return state

    def assert_xy(state, moves, xy, chnl):
        T = len(moves[xy])
        for t in range(T):
            for i, j in moves[xy][:t+1]:
                assert state[chnl[T - t - 1], i, j] == 1

    for n in range(16):
        state = step(state)
        x, y = y, x
        assert_xy(state, moves, x, govars.Xt_CHNL)
        assert_xy(state, moves, y, govars.Yt_CHNL)

    print(state)
    print(moves)



def random_action2d(state):
    invalid_moves = state[govars.INVD_CHNL].flatten()
    #invalid_moves = np.append(invalid_moves, 0)
    move_weights = 1 - invalid_moves
    action1d = gogame.random_weighted_action(move_weights)
    state = gogame.next_state(state, action1d)
    action2d = action1d // state.shape[-1], action1d % state.shape[-1]
    return action2d, state
