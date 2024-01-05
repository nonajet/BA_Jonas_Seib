import logging

from Daten.Visualisierung.mylib import get_dog_id

paw_order = []


def save_paws(dog_paws):
    assert dog_paws
    log_namestring = get_dog_id()
    logger = logging.getLogger(log_namestring)

    for paw_obj in dog_paws:
        if paw_obj:
            logger.info(paw_obj.name)
            logger.info(paw_obj.ground)
            logger.info(paw_obj.lastContact)
            logger.info(paw_obj.set_since)
            logger.info(paw_obj.global_pos)
            # logger.info(paw_obj.area)

    # TODO: next
    active_paws = [paw for paw in dog_paws if paw.ground]
    paw_order.append(active_paws)


print(paw_order)
