import mujoco
import mujoco.viewer
import time

from __init__ import *


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            with viewer.lock():
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.perf_counter() - start)))
