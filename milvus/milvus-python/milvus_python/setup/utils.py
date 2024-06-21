import shutil
import logging
import os

def set_up_log(index_name) -> None:
    logger = logging.getLogger(index_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(f"./milvus_python/logs/setup/{index_name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stdout_handler)
    return logger


def remove_volume_folder():
    try:
        folder_path = "/Users/joaobaptista/Documents/personal/banco-de-dados-massivos/milvus/volumes"
        shutil.rmtree(folder_path)
        print("Folder and its content removed")
    except Exception as e:
        print("Folder not deleted", e)
