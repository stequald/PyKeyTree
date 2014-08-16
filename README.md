PyKeyTree
===========

PyKeyTree is a python implementation of [KeyTree](https://github.com/stequald/KeyTree). PyKeyTree requires the python ecdsa module. You can install it by doing `pip install ecdsa`.

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
    bfe6ee6af1d8dc0dedbb64fb257cfd3b24a5b2245d65fdc9fe74aa85eeba5482
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

###### To print data on all the nodes in the chain, use the all option:
    ./kt.py --all
    ./kt.py -a

###### It is also possible to print data on the nodes in a different order:
    ./kt.py --traverse levelorder
    ./kt.py -sh -trav postorder
    ./kt.py -ek -trav preorder

###### For more information on each node use the verbose option:
    ./kt.py --verbose
    ./kt.py -v
    
###### There is also the testnet option:
    ./kt.py --testnet
    ./kt.py -tn

###### Use the no echo option to not echo your inputs:
    ./kt.py --noecho
    ./kt.py -ne

##### For more information on how to use KeyTree do:
    ./kt.py --help