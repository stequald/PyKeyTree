PyKeyTree
===========

PyKeyTree is a python implementation of [KeyTree](https://github.com/stequald/KeyTree). PyKeyTree requires the python ecdsa module. You can install it by doing `pip install ecdsa`.

#### How to use:

The input seed is coverted to hex before it used as the master seed, or you can also use the --seedhex option to enter the seed in hex directly. Extended Keys can be in hex or base58. Examples below.

###### To use KeyTree simply do the following:
    ./kt.py
    Enter Seed:
    correct horse battery staple
    Enter Chain:
    0'/0

###### Use the BIP39 option to check if the seed/mnemonic conforms to bip39 rules:
    ./kt.py --bip39
    Enter Seed:
    pilot dolphin motion portion survey sock turkey afford destroy knee sock sibling
    Enter Chain:
    0'/0

###### Use the hex option to enter the seed in hex:
    ./kt.py --seedhex 
    Enter Seed in Hex:
    7b1f95ed9a1c9319172c2dd4cc765fb82bad1a3be1cfc89fc37006c0dbbcbe3d
    Enter Chain:
    0'/1/2
  
###### Use the hash seed option to do a number of specific rounds of sha256 on your seed. If the bip39 option is used hash seed option will be ignored:
    ./kt.py --seedhex --hashseed
    Enter Seed in Hex:
    7b1f95ed9a1c9319172c2dd4cc765fb82bad1a3be1cfc89fc37006c0dbbcbe3d
    Enter Chain:
    0'/1/2
    Enter number of rounds of Sha256 hash:
    2

###### Use the generate BIP39 mnemonic option to generate a mnemonic using os.urandom:
    ./kt.py --generateBIP39mnemonic
    ./kt.py -gbip39m

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

###### You can specify all options at once with the no prompt option. But it is discouraged because on most OS commands are stored in a history file:
    ./kt.py --noprompt -s "this is a password" --chain "(0-1)'/(6-8)'" -trav levelorder
    ./kt.py -np -s "this is a password" -c "(0-1)'/(6-8)'" -hs 3 -v
    ./kt.py -np --extkey xprv9uHRZZhk6KAJC1avXpDAp4MDc3sQKNxDiPvvkX8Br5ngLNv1TxvUxt4cV1rGL5hj6KCesnDYUhd7oWgT11eZG7XnxHrnYeSvkzY7d2bhkJ7 -c "(0-1)'/8"
    ./kt.py -np --b39 -s "pilot dolphin motion portion survey sock turkey afford destroy knee sock sibling" -c "44'/0'/(0-1)'"
    ./kt.py -np -sh 936ae011512b96e7ce3ff05d464e3801834d023249baabfebfe13e593dc33610ea68279c271df6bab7cfbea8bbcf470e050fe6589f552f7e1f6c80432c7bcc57 -c "44'/0'/(0-1)'"


##### For more information on how to use KeyTree do:
    ./kt.py --help
