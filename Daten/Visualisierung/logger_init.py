import logging


def init_logger(logname="default.log"):
    logging.basicConfig(filename=logname,
                        filemode='w',  # w:=truncate+write; a:=append
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
