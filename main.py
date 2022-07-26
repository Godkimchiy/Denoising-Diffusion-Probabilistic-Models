import traceback
import logging
import yaml
import sys
import os

from configs.parser import parse_args_and_config
from runners.diffusion import Diffusion


def main():
    args, configs = parse_args_and_config() # args는 이 파일에서만 쓰는 인자인가?
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp. instance id = {}".format(os.getpid()))
    logging.info("Exp. comment = {}".format(args.comment)) # ? 어떤 용도일까

    try:
        runner = Diffusion(args, config) # Diffusion model class 선언

        if args.sample: # case(1): generate images
            runner.sample()
        elif args.test: # case(2): test models
            runner.test()
        else:           # case(3): train models
            runner.train()
    
    except Exception:
        loggin.error(traceback.format_exc()) # 몰루
    
    return 0



if __name__=="__main__":
    sys.exit(main())
