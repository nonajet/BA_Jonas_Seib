import copy
import logging

from Daten.Visualisierung.mylib import get_dog_log

raw_paw_order = []


def save_paws(dog_paws):
    assert dog_paws
    log_namestring = get_dog_log()
    logger = logging.getLogger(log_namestring)

    for paw_obj in dog_paws:
        if paw_obj:
            logger.info(paw_obj.name)
            logger.info(paw_obj.ground)
            logger.info(paw_obj.lastContact)
            logger.info(paw_obj.set_since)
            logger.info(paw_obj.global_pos)
            # logger.info(paw_obj.area)

    active_paws = [copy.copy(paw) for paw in dog_paws if paw.ground]
    raw_paw_order.append(active_paws)


def paw_validation():  # mit list
    UNINIT_VAL = ''  # default value for not yet initialised paws (order)
    t_step = 1  # time step
    tot_steps = 0  # total no. of steps
    paws_in_order = [''] * 4
    doubtful_paws = {}
    # TODO maybe with (unique) queue?
    # planted & lifted both in comparison to time step before
    for paws_on_ground in raw_paw_order:
        paws_planted, paws_lifted = get_changed_paws(t_step)
        if UNINIT_VAL in paws_in_order:  # some yet uninit. values for paw order -> set order first
            for new_paw in paws_planted:
                paw_turn = tot_steps % 4
                if paws_in_order[paw_turn] == UNINIT_VAL:
                    paws_in_order[paw_turn] = new_paw
                else:  # paw already had order no. assigned
                    doubtful_paws[t_step] = new_paw
                    paws_in_order[paw_turn] = UNINIT_VAL
                tot_steps += 1
        else:
            ex_next_paw = paws_in_order[tot_steps % 4]
            if ex_next_paw in paws_planted:
                tot_steps += len(paws_planted)
            else:
                print('problem')
        t_step += 1


# mit dict
# def paw_validation():
#     UNINIT_VAL = -1  # default value for not yet initialised paws
#     t_step = 1  # time step
#     tot_steps = 0  # total no. of steps
#     paw_dict = {'fl': UNINIT_VAL, 'fr': UNINIT_VAL, 'bl': UNINIT_VAL, 'br': UNINIT_VAL}
#     paws_in_order = [''] * 4
#     doubtful_paws = {}
#     # planted & lifted both in comparison to time step before
#     for paws_on_ground in raw_paw_order:
#         paws_planted, paws_lifted = get_changed_paws(t_step)
#         if UNINIT_VAL in paw_dict.values():  # some yet uninit. values for paw order -> set order first
#             for new_paw in paws_planted:
#                 if paw_dict[new_paw] == UNINIT_VAL:
#                     paw_dict[new_paw] = tot_steps % 4
#                 else:  # paw already had order no. assigned
#                     doubtful_paws[t_step] = new_paw
#                     paw_dict[new_paw] = UNINIT_VAL
#                 tot_steps += 1
#         else:
#             ex_next_paw = paw_dict[tot_steps % 4]
#             if ex_next_paw in paws_planted:
#                 tot_steps += len(paws_planted)
#             else:
#                 print('problem')
#         t_step += 1

def get_changed_paws(t_step):
    try:
        prev_active_paws = {paw.name for paw in raw_paw_order[t_step - 1]}
        active_paws = {paw.name for paw in raw_paw_order[t_step]}
        if t_step <= 1:
            return list(active_paws), []  # very first paws to touch mat are newly planted of course
        else:
            lifted_paws = list(prev_active_paws - active_paws)
            planted_paws = list(active_paws - prev_active_paws)
            return planted_paws, lifted_paws
    except IndexError:
        return [], []
