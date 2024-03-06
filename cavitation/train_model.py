import datetime
import logging
import os
from io import StringIO

import hydra
import omegaconf
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

from cavitation.commiter.commiter import commit_experiment
from cavitation.data.get_data import get_data
from cavitation.logger.easy_logger import get_logger
from cavitation.models.model import get_model


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    # initializing
    time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    exp_commit_message, exp_commit_hash = commit_experiment()

    logger = get_logger(__name__)

    wandb_name = datetime.datetime.strptime(time_tag, "%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M:%S")
    # USER: change this line respecting your application
    wandb.init(entity=..., project="cavitation", name=wandb_name)
    wandb.config.update(omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    wandb.config.update({"exp_commit_message": exp_commit_message, "exp_commit_hash": exp_commit_hash})

    (train_x, train_y), (test_x, test_y) = get_data()   # USER: change this line respecting your application
    model = get_model()

    # model summary
    logger.info("model summary")
    with StringIO() as buf:
        model.summary(print_fn=lambda x: buf.write(x + '\n'))
        summary_str = buf.getvalue()
    logger.info(summary_str)

    # training
    logger.info("start training")

    if len(tf.config.list_physical_devices('GPU')) > 0:
        logger.info("GPU is available")
    else:
        logger.info("GPU is not available")

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=opt, loss=..., metrics=...)     # USER: change this line respecting your application

    best_model_path = "models/{time}/model_best.keras".format(time=time_tag)
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(best_model_path, save_best_only=True, verbose=0)
    wandb_callback = WandbCallback(save_model=False, )      # USER: change this line respecting your application
    callbacks = [wandb_callback, best_model_callback]

    # USER: change this line respecting your application
    model.fit(train_x, train_y, epochs=cfg.epochs, batch_size=cfg.batch_size, validation_data=(test_x, test_y), callbacks=callbacks)

    # save the last and best model
    model_last_path = "models/{time}/model_last.keras".format(time=time_tag)
    model.save(model_last_path)
    logger.info("model saved to {}".format(model_last_path))

    last_model_artifact = wandb.Artifact("last_model", type="model")
    last_model_artifact.add_file(model_last_path)
    wandb.log_artifact(last_model_artifact)

    best_model_artifact = wandb.Artifact("best_model", type="model")
    best_model_artifact.add_file(best_model_path)
    wandb.log_artifact(best_model_artifact)

    # evaluate the last and best model
    model_best = tf.keras.models.load_model(best_model_path)
    logger.info("evaluating the best model:")
    loss_last, acc_last = model_best.evaluate(test_x, test_y, verbose=0)    # USER: change this line respecting your application
    logger.info("loss: {:.6f}, acc: {:.6f}".format(loss_last, acc_last))    # USER: change this line respecting your application

    logger.info("evaluating the last model:")
    loss_last, acc_last = model.evaluate(test_x, test_y, verbose=0)         # USER: change this line respecting your application
    logger.info("loss: {:.6f}, acc: {:.6f}".format(loss_last, acc_last))    # USER: change this line respecting your application

    wandb.finish()


if __name__ == "__main__":
    # Info: environment variable 'TF_CPP_MIN_LOG_LEVEL' has been set to '2' in the Makefile `setup_project` target
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ["WANDB_SILENT"] = "true"

    main()
