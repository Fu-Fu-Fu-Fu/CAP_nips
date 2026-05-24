from .alkox_emulator import AlkoxEmulatorTask
from .ackley_family import AckleyFamilyTask
from .benzylation_emulator import BenzylationEmulatorTask
from .branin_family import BraninFamilyTask
from .hplc_emulator import HplcEmulatorTask
from .goldstein_price import GoldsteinPriceTask
from .goldstein_price_family import GoldsteinPriceFamilyTask
from .hartmann_3d import Hartmann3DTask
from .hartmann_3d_family import Hartmann3DFamilyTask
from .hartmann_6d import Hartmann6DTask
from .hartmann_6d_family import Hartmann6DFamilyTask
from .registry import register_task

register_task(AlkoxEmulatorTask())
register_task(AckleyFamilyTask(dim=5))
register_task(AckleyFamilyTask(dim=10))
register_task(BenzylationEmulatorTask())
register_task(BraninFamilyTask())
register_task(GoldsteinPriceTask())
register_task(GoldsteinPriceFamilyTask())
register_task(Hartmann3DTask())
register_task(Hartmann3DFamilyTask())
register_task(Hartmann6DTask())
register_task(Hartmann6DFamilyTask())
register_task(HplcEmulatorTask())
