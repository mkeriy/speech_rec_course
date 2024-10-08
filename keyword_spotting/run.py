import hydra
import omegaconf
from pytorch_lightning import seed_everything

from src.module import KWS, KWS_resnet
from utils import omegaconf_extension

from pytorch_lightning.loggers import WandbLogger


@omegaconf_extension
@hydra.main(version_base="1.2", config_path="conf", config_name="bcresnet.yaml")
def main(conf: omegaconf.DictConfig) -> None:
    seed_everything(121, workers=True)
    logger = WandbLogger(log_model="all", project="KWS", entity="malkevich-dim")
    module = KWS_resnet(conf)
    
    trainer = hydra.utils.instantiate(conf.trainer, logger=logger)
    trainer.fit(module)


if __name__ == "__main__":
    main()
