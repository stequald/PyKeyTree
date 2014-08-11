PyKeyTree
===========

PyKeyTree is a python implementation of [KeyTree](https://github.com/stequald/KeyTree).

#### How to use:

Extended Keys can be in hex or base58. Seed can be in ASCII or hex. Examples below.

###### To use KeyTree simply do the following:
    ./kt.py
    Enter Seed:
    correct horse battery staple
    Enter Chain:
    0'/0

###### Use the hex option to enter the seed in hex:
    ./kt.py --seedhex 
    Enter Seed in Hex:
    5e8fc96380121030d090dc2242c4c4159235cdb9fb027e2fa27cdcf0f7216129
    Enter Chain:
    0'/1/2
  
###### Use the extended key option to enter the extended key in lieu of the seed:
    ./kt.py --extkey 
    ./kt.py -ek 

###### It is also possible to print multiple chain paths together:
    ./kt.py -ek
    Enter Extended Key:
    xprv9uHRZZhk6KAJC1avXpDAp4MDc3sQKNxDiPvvkX8Br5ngLNv1TxvUxt4cV1rGL5hj6KCesnDYUhd7oWgT11eZG7XnxHrnYeSvkzY7d2bhkJ7
    Enter Chain:
    0'/(3-6)'/(1-2)/8

###### To output all the node data on the chain, use the all option:
    ./kt.py --all
    ./kt.py -a

###### It is also possible to output the nodes in a different order:
    ./kt.py --traverse levelorder
    ./kt.py -sh -trav postorder
    ./kt.py -ek -trav preorder

###### For more information on nodes use the verbose option:
    ./kt.py --verbose
    ./kt.py -v

##### For more information on how to use KeyTree do:
    ./kt.py --help