from AddRank import *
from bleuRank import *
from EncodeRank import *
from HredEncodeRank import *
from HredRank import *

from dataprocess.unit import *
# baserank = HredEncodeRank()
baserank = HredRank()
baserank.set('../data/AddRank.set')

myset = UnitSet();
newunit = Unit()
newunit.context = baserank.unitset[20].context
myset.allunit = baserank.unitset[:30]

newunit.context = newunit.context[:-1]
print(newunit.context)
print(myset.allunit[20])

check = baserank.unitset[:10]
print(baserank.distance(newunit,myset.allunit[20]))
print(baserank.search(newunit,ascending=False))