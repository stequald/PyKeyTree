#!/usr/bin/env python

# from http://eli.thegreenplace.net/2009/03/07/computing-modular-square-roots-in-python/

def modular_sqrt(a, p):
    """ Find a quadratic residue (mod p) of 'a'. p
    must be an odd prime.
    
    Solve the congruence of the form:
    x^2 = a (mod p)
    And returns x. Note that p - x is also a root.
    
    0 is returned is no square root exists for
    these a and p.
    
    The Tonelli-Shanks algorithm is used (except
    for some simple cases in which the solution
    is known from an identity). This algorithm
    runs in polynomial time (unless the
    generalized Riemann hypothesis is false).
    """
    # Simple cases
    #
    if legendre_symbol(a, p) != 1:
        return 0
    elif a == 0:
        return 0
    elif p == 2:
        return p
    elif p % 4 == 3:
        return pow(a, (p + 1) / 4, p)
    
    # Partition p-1 to s * 2^e for an odd s (i.e.
    # reduce all the powers of 2 from p-1)
    #
    s = p - 1
    e = 0
    while s % 2 == 0:
        s /= 2
        e += 1
        
    # Find some 'n' with a legendre symbol n|p = -1.
    # Shouldn't take long.
    #
    n = 2
    while legendre_symbol(n, p) != -1:
        n += 1
        
    # Here be dragons!
    # Read the paper "Square roots from 1; 24, 51,
    # 10 to Dan Shanks" by Ezra Brown for more
    # information
    #
    
    # x is a guess of the square root that gets better
    # with each iteration.
    # b is the "fudge factor" - by how much we're off
    # with the guess. The invariant x^2 = ab (mod p)
    # is maintained throughout the loop.
    # g is used for successive powers of n to update
    # both a and b
    # r is the exponent - decreases with each update
    #
    x = pow(a, (s + 1) / 2, p)
    b = pow(a, s, p)
    g = pow(n, s, p)
    r = e
    
    while True:
        t = b
        m = 0
        for m in xrange(r):
            if t == 1:
                break
            t = pow(t, 2, p)
            
        if m == 0:
            return x
        
        gs = pow(g, 2 ** (r - m - 1), p)
        g = (gs * gs) % p
        x = (x * gs) % p
        b = (b * g) % p
        r = m
        
def legendre_symbol(a, p):
    """ Compute the Legendre symbol a|p using
    Euler's criterion. p is a prime, a is
    relatively prime to p (if p divides
    a, then a|p = 0)
    
    Returns 1 if a has a square root modulo
    p, -1 otherwise.
    """
    ls = pow(a, (p - 1) / 2, p)
    return -1 if ls == p - 1 else ls

# much of the code is 'borrowed' from electrum,
# https://gitorious.org/electrum/electrum
# and is under the GPLv3.

import hashlib, base64, ecdsa, re
import hmac

def rev_hex(s):
    return s.decode('hex')[::-1].encode('hex')

def int_to_hex(i, length=1):
    s = hex(i)[2:].rstrip('L')
    s = "0"*(2*length - len(s)) + s
    return rev_hex(s)

def var_int(i):
    # https://en.bitcoin.it/wiki/Protocol_specification#Variable_length_integer
    if i<0xfd:
        return int_to_hex(i)
    elif i<=0xffff:
        return "fd"+int_to_hex(i,2)
    elif i<=0xffffffff:
        return "fe"+int_to_hex(i,4)
    else:
        return "ff"+int_to_hex(i,8)


def sha256(x):
    return hashlib.sha256(x).digest()

def Hash(x):
    if type(x) is unicode: x=x.encode('utf-8')
    return sha256(sha256(x))

# pywallet openssl private key implementation

def i2d_ECPrivateKey(pkey, compressed=False):
    if compressed:
        key = '3081d30201010420' + \
              '%064x' % pkey.secret + \
              'a081a53081a2020101302c06072a8648ce3d0101022100' + \
              '%064x' % _p + \
              '3006040100040107042102' + \
              '%064x' % _Gx + \
              '022100' + \
              '%064x' % _r + \
              '020101a124032200'
    else:
        key = '308201130201010420' + \
              '%064x' % pkey.secret + \
              'a081a53081a2020101302c06072a8648ce3d0101022100' + \
              '%064x' % _p + \
              '3006040100040107044104' + \
              '%064x' % _Gx + \
              '%064x' % _Gy + \
              '022100' + \
              '%064x' % _r + \
              '020101a144034200'
        
    return key.decode('hex') + i2o_ECPublicKey(pkey.pubkey, compressed)
    
def i2o_ECPublicKey(pubkey, compressed=False):
    # public keys are 65 bytes long (520 bits)
    # 0x04 + 32-byte X-coordinate + 32-byte Y-coordinate
    # 0x00 = point at infinity, 0x02 and 0x03 = compressed, 0x04 = uncompressed
    # compressed keys: <sign> <x> where <sign> is 0x02 if y is even and 0x03 if y is odd
    if compressed:
        if pubkey.point.y() & 1:
            key = '03' + '%064x' % pubkey.point.x()
        else:
            key = '02' + '%064x' % pubkey.point.x()
    else:
        key = '04' + \
              '%064x' % pubkey.point.x() + \
              '%064x' % pubkey.point.y()
            
    return key.decode('hex')
            
# end pywallet openssl private key implementation

                                                
            
############ functions from pywallet ##################### 

def hash_160(public_key):
    try:
        md = hashlib.new('ripemd160')
        md.update(sha256(public_key))
        return md.digest()
    except Exception:
        import ripemd
        md = ripemd.new(sha256(public_key))
        return md.digest()


def public_key_to_bc_address(public_key):
    h160 = hash_160(public_key)
    return hash_160_to_bc_address(h160)

def hash_160_to_bc_address(h160, addrtype = 0):
    vh160 = chr(addrtype) + h160
    h = Hash(vh160)
    addr = vh160 + h[0:4]
    return b58encode(addr)

def bc_address_to_hash_160(addr):
    bytes = b58decode(addr, 25)
    return ord(bytes[0]), bytes[1:21]


__b58chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
__b58base = len(__b58chars)

def b58encode(v):
    """ encode v, which is a string of bytes, to base58."""

    long_value = 0L
    for (i, c) in enumerate(v[::-1]):
        long_value += (256**i) * ord(c)

    result = ''
    while long_value >= __b58base:
        div, mod = divmod(long_value, __b58base)
        result = __b58chars[mod] + result
        long_value = div
    result = __b58chars[long_value] + result

    # Bitcoin does a little leading-zero-compression:
    # leading 0-bytes in the input become leading-1s
    nPad = 0
    for c in v:
        if c == '\0': nPad += 1
        else: break

    return (__b58chars[0]*nPad) + result

def b58decode(v, length):
    """ decode v into a string of len bytes."""
    long_value = 0L
    for (i, c) in enumerate(v[::-1]):
        long_value += __b58chars.find(c) * (__b58base**i)

    result = ''
    while long_value >= 256:
        div, mod = divmod(long_value, 256)
        result = chr(mod) + result
        long_value = div
    result = chr(long_value) + result

    nPad = 0
    for c in v:
        if c == __b58chars[0]: nPad += 1
        else: break

    result = chr(0)*nPad + result
    if length is not None and len(result) != length:
        return None

    return result


def EncodeBase58Check(vchIn):
    hash = Hash(vchIn)
    return b58encode(vchIn + hash[0:4])

def DecodeBase58Check(psz):
    vchRet = b58decode(psz, None)
    key = vchRet[0:-4]
    csum = vchRet[-4:]
    hash = Hash(key)
    cs32 = hash[0:4]
    if cs32 != csum:
        return None
    else:
        return key

def PrivKeyToSecret(privkey):
    return privkey[9:9+32]

def SecretToASecret(secret, compressed=False, addrtype=0):
    vchIn = chr((addrtype+128)&255) + secret
    if compressed: vchIn += '\01'
    return EncodeBase58Check(vchIn)

def ASecretToSecret(key, addrtype=0):
    vch = DecodeBase58Check(key)
    if vch and vch[0] == chr((addrtype+128)&255):
        return vch[1:]
    else:
        return False

def regenerate_key(sec):
    b = ASecretToSecret(sec)
    if not b:
        return False
    b = b[0:32]
    return EC_KEY(b)

def GetPubKey(pubkey, compressed=False):
    return i2o_ECPublicKey(pubkey, compressed)

def GetPrivKey(pkey, compressed=False):
    return i2d_ECPrivateKey(pkey, compressed)

def GetSecret(pkey):
    return ('%064x' % pkey.secret).decode('hex')

def is_compressed(sec):
    b = ASecretToSecret(sec)
    return len(b) == 33


def public_key_from_private_key(sec):
    # rebuild public key from private key, compressed or uncompressed
    pkey = regenerate_key(sec)
    assert pkey
    compressed = is_compressed(sec)
    public_key = GetPubKey(pkey.pubkey, compressed)
    return public_key.encode('hex')


def address_from_private_key(sec):
    public_key = public_key_from_private_key(sec)
    address = public_key_to_bc_address(public_key.decode('hex'))
    return address


def is_valid(addr):
    ADDRESS_RE = re.compile('[1-9A-HJ-NP-Za-km-z]{26,}\\Z')
    if not ADDRESS_RE.match(addr): return False
    try:
        addrtype, h = bc_address_to_hash_160(addr)
    except Exception:
        return False
    return addr == hash_160_to_bc_address(h, addrtype)


########### end pywallet functions #######################

try:
    from ecdsa.ecdsa import curve_secp256k1, generator_secp256k1
except Exception:
    print "cannot import ecdsa.curve_secp256k1. You probably need to upgrade ecdsa.\nTry: sudo pip install --upgrade ecdsa"
    exit()

from ecdsa.curves import SECP256k1
from ecdsa.ellipticcurve import Point
from ecdsa.util import string_to_number, number_to_string

def msg_magic(message):
    varint = var_int(len(message))
    encoded_varint = "".join([chr(int(varint[i:i+2], 16)) for i in xrange(0, len(varint), 2)])
    return "\x18Bitcoin Signed Message:\n" + encoded_varint + message


def verify_message(address, signature, message):
    try:
        EC_KEY.verify_message(address, signature, message)
        return True
    except Exception as e:
        print "error: Verification error: {0}".format(e)
        return False


def encrypt_message(message, pubkey):
    return EC_KEY.encrypt_message(message, pubkey.decode('hex'))


def chunks(l, n):
    return [l[i:i+n] for i in xrange(0, len(l), n)]


def ECC_YfromX(x,curved=curve_secp256k1, odd=True):
    _p = curved.p()
    _a = curved.a()
    _b = curved.b()
    for offset in range(128):
        Mx = x + offset
        My2 = pow(Mx, 3, _p) + _a * pow(Mx, 2, _p) + _b % _p
        My = pow(My2, (_p+1)/4, _p )

        if curved.contains_point(Mx,My):
            if odd == bool(My&1):
                return [My,offset]
            return [_p-My,offset]
    raise Exception('ECC_YfromX: No Y found')

def private_header(msg,v):
    assert v<1, "Can't write version %d private header"%v
    r = ''
    if v==0:
        r += ('%08x'%len(msg)).decode('hex')
        r += sha256(msg)[:2]
    return ('%02x'%v).decode('hex') + ('%04x'%len(r)).decode('hex') + r

def public_header(pubkey,v):
    assert v<1, "Can't write version %d public header"%v
    r = ''
    if v==0:
        r = sha256(pubkey)[:2]
    return '\x6a\x6a' + ('%02x'%v).decode('hex') + ('%04x'%len(r)).decode('hex') + r


def negative_point(P):
    return Point( P.curve(), P.x(), -P.y(), P.order() )


def point_to_ser(P, comp=True ):
    if comp:
        return ( ('%02x'%(2+(P.y()&1)))+('%064x'%P.x()) ).decode('hex')
    return ( '04'+('%064x'%P.x())+('%064x'%P.y()) ).decode('hex')


def ser_to_point(Aser):
    curve = curve_secp256k1
    generator = generator_secp256k1
    _r  = generator.order()
    assert Aser[0] in ['\x02','\x03','\x04']
    if Aser[0] == '\x04':
        return Point( curve, str_to_long(Aser[1:33]), str_to_long(Aser[33:]), _r )
    Mx = string_to_number(Aser[1:])
    return Point( curve, Mx, ECC_YfromX(Mx, curve, Aser[0]=='\x03')[0], _r )



class EC_KEY(object):
    def __init__( self, k ):
        secret = string_to_number(k)
        self.pubkey = ecdsa.ecdsa.Public_key( generator_secp256k1, generator_secp256k1 * secret )
        self.privkey = ecdsa.ecdsa.Private_key( self.pubkey, secret )
        self.secret = secret

    def get_public_key(self, compressed=True):
        return point_to_ser(self.pubkey.point, compressed).encode('hex')

    def sign_message(self, message, compressed, address):
        private_key = ecdsa.SigningKey.from_secret_exponent( self.secret, curve = SECP256k1 )
        public_key = private_key.get_verifying_key()
        signature = private_key.sign_digest_deterministic( Hash( msg_magic(message) ), hashfunc=hashlib.sha256, sigencode = ecdsa.util.sigencode_string )
        assert public_key.verify_digest( signature, Hash( msg_magic(message) ), sigdecode = ecdsa.util.sigdecode_string)
        for i in range(4):
            sig = base64.b64encode( chr(27 + i + (4 if compressed else 0)) + signature )
            try:
                self.verify_message( address, sig, message)
                return sig
            except Exception:
                continue
        else:
            raise Exception("error: cannot sign message")


    @classmethod
    def verify_message(self, address, signature, message):
        """ See http://www.secg.org/download/aid-780/sec1-v2.pdf for the math """
        from ecdsa import numbertheory, util
        curve = curve_secp256k1
        G = generator_secp256k1
        order = G.order()
        # extract r,s from signature
        sig = base64.b64decode(signature)
        if len(sig) != 65: raise Exception("Wrong encoding")
        r,s = util.sigdecode_string(sig[1:], order)
        nV = ord(sig[0])
        if nV < 27 or nV >= 35:
            raise Exception("Bad encoding")
        if nV >= 31:
            compressed = True
            nV -= 4
        else:
            compressed = False

        recid = nV - 27
        # 1.1
        x = r + (recid/2) * order
        # 1.3
        alpha = ( x * x * x  + curve.a() * x + curve.b() ) % curve.p()
        beta = modular_sqrt(alpha, curve.p())
        y = beta if (beta - recid) % 2 == 0 else curve.p() - beta
        # 1.4 the constructor checks that nR is at infinity
        R = Point(curve, x, y, order)
        # 1.5 compute e from message:
        h = Hash( msg_magic(message) )
        e = string_to_number(h)
        minus_e = -e % order
        # 1.6 compute Q = r^-1 (sR - eG)
        inv_r = numbertheory.inverse_mod(r,order)
        Q = inv_r * ( s * R + minus_e * G )
        public_key = ecdsa.VerifyingKey.from_public_point( Q, curve = SECP256k1 )
        # check that Q is the public key
        public_key.verify_digest( sig[1:], h, sigdecode = ecdsa.util.sigdecode_string)
        # check that we get the original signing address
        addr = public_key_to_bc_address( point_to_ser(public_key.pubkey.point, compressed) )
        if address != addr:
            raise Exception("Bad signature")


    # ecdsa encryption/decryption methods
    # credits: jackjack, https://github.com/jackjack-jj/jeeq

    @classmethod
    def encrypt_message(self, message, pubkey):
        generator = generator_secp256k1
        curved = curve_secp256k1
        r = ''
        msg = private_header(message,0) + message
        msg = msg + ('\x00'*( 32-(len(msg)%32) ))
        msgs = chunks(msg,32)

        _r  = generator.order()
        str_to_long = string_to_number

        P = generator
        if len(pubkey)==33: #compressed
            pk = Point( curve_secp256k1, str_to_long(pubkey[1:33]), ECC_YfromX(str_to_long(pubkey[1:33]), curve_secp256k1, pubkey[0]=='\x03')[0], _r )
        else:
            pk = Point( curve_secp256k1, str_to_long(pubkey[1:33]), str_to_long(pubkey[33:65]), _r )

        for i in range(len(msgs)):
            n = ecdsa.util.randrange( pow(2,256) )
            Mx = str_to_long(msgs[i])
            My, xoffset = ECC_YfromX(Mx, curved)
            M = Point( curved, Mx+xoffset, My, _r )
            T = P*n
            U = pk*n + M
            toadd = point_to_ser(T) + point_to_ser(U)
            toadd = chr(ord(toadd[0])-2 + 2*xoffset) + toadd[1:]
            r += toadd

        return base64.b64encode(public_header(pubkey,0) + r)


    def decrypt_message(self, enc):
        G = generator_secp256k1
        curved = curve_secp256k1
        pvk = self.secret
        pubkeys = [point_to_ser(G*pvk,True), point_to_ser(G*pvk,False)]
        enc = base64.b64decode(enc)
        str_to_long = string_to_number

        assert enc[:2]=='\x6a\x6a'

        phv = str_to_long(enc[2])
        assert phv==0, "Can't read version %d public header"%phv
        hs = str_to_long(enc[3:5])
        public_header=enc[5:5+hs]
        checksum_pubkey=public_header[:2]
        address=filter(lambda x:sha256(x)[:2]==checksum_pubkey, pubkeys)
        assert len(address)>0, 'Bad private key'
        address=address[0]
        enc=enc[5+hs:]
        r = ''
        for Tser,User in map(lambda x:[x[:33],x[33:]], chunks(enc,66)):
            ots = ord(Tser[0])
            xoffset = ots>>1
            Tser = chr(2+(ots&1))+Tser[1:]
            T = ser_to_point(Tser)
            U = ser_to_point(User)
            V = T*pvk
            Mcalc = U + negative_point(V)
            r += ('%064x'%(Mcalc.x()-xoffset)).decode('hex')

        pvhv = str_to_long(r[0])
        assert pvhv==0, "Can't read version %d private header"%pvhv
        phs = str_to_long(r[1:3])
        private_header = r[3:3+phs]
        size = str_to_long(private_header[:4])
        checksum = private_header[4:6]
        r = r[3+phs:]

        msg = r[:size]
        hashmsg = sha256(msg)[:2]
        checksumok = hashmsg==checksum

        return [msg, checksumok, address]





###################################### BIP32 ##############################

random_seed = lambda n: "%032x"%ecdsa.util.randrange( pow(2,n) )
BIP32_PRIME = 0x80000000

def bip32_init(seed):
    import hmac
    seed = seed.decode('hex')        
    I = hmac.new("Bitcoin seed", seed, hashlib.sha512).digest()

    master_secret = I[0:32]
    master_chain = I[32:]

    K, K_compressed = get_pubkeys_from_secret(master_secret)
    return master_secret, master_chain, K, K_compressed


def get_pubkeys_from_secret(secret):
    # public key
    private_key = ecdsa.SigningKey.from_string( secret, curve = SECP256k1 )
    public_key = private_key.get_verifying_key()
    K = public_key.to_string()
    K_compressed = GetPubKey(public_key.pubkey,True)
    return K, K_compressed



# Child private key derivation function (from master private key)
# k = master private key (32 bytes)
# c = master chain code (extra entropy for key derivation) (32 bytes)
# n = the index of the key we want to derive. (only 32 bits will be used)
# If n is negative (i.e. the 32nd bit is set), the resulting private key's
#  corresponding public key can NOT be determined without the master private key.
# However, if n is positive, the resulting private key's corresponding
#  public key can be determined without the master private key.
def CKD(k, c, n):
    import hmac
    from ecdsa.util import string_to_number, number_to_string
    order = generator_secp256k1.order()
    keypair = EC_KEY(k)
    K = GetPubKey(keypair.pubkey,True)

    if n & BIP32_PRIME: # We want to make a "secret" address that can't be determined from K
        data = chr(0) + k + rev_hex(int_to_hex(n,4)).decode('hex')
        I = hmac.new(c, data, hashlib.sha512).digest()
    else: # We want a "non-secret" address that can be determined from K
        I = hmac.new(c, K + rev_hex(int_to_hex(n,4)).decode('hex'), hashlib.sha512).digest()
        
    k_n = number_to_string( (string_to_number(I[0:32]) + string_to_number(k)) % order , order )
    c_n = I[32:]
    return k_n, c_n

# Child public key derivation function (from public key only)
# K = master public key 
# c = master chain code
# n = index of key we want to derive
# This function allows us to find the nth public key, as long as n is 
#  non-negative. If n is negative, we need the master private key to find it.
def CKD_prime(K, c, n):
    import hmac
    from ecdsa.util import string_to_number, number_to_string
    order = generator_secp256k1.order()

    if n & BIP32_PRIME: raise

    K_public_key = ecdsa.VerifyingKey.from_string( K, curve = SECP256k1 )
    K_compressed = GetPubKey(K_public_key.pubkey,True)

    I = hmac.new(c, K_compressed + rev_hex(int_to_hex(n,4)).decode('hex'), hashlib.sha512).digest()

    curve = SECP256k1
    pubkey_point = string_to_number(I[0:32])*curve.generator + K_public_key.pubkey.point
    public_key = ecdsa.VerifyingKey.from_public_point( pubkey_point, curve = SECP256k1 )

    K_n = public_key.to_string()
    K_n_compressed = GetPubKey(public_key.pubkey,True)
    c_n = I[32:]

    return K_n, K_n_compressed, c_n



def bip32_private_derivation(k, c, branch, sequence):
    assert sequence.startswith(branch)
    sequence = sequence[len(branch):]
    for n in sequence.split('/'):
        if n == '': continue
        n = int(n[:-1]) + BIP32_PRIME if n[-1] == "'" else int(n)
        k, c = CKD(k, c, n)
    K, K_compressed = get_pubkeys_from_secret(k)
    return k.encode('hex'), c.encode('hex'), K.encode('hex'), K_compressed.encode('hex')


def bip32_public_derivation(c, K, branch, sequence):
    assert sequence.startswith(branch)
    sequence = sequence[len(branch):]
    for n in sequence.split('/'):
        n = int(n)
        K, cK, c = CKD_prime(K, c, n)

    return c.encode('hex'), K.encode('hex'), cK.encode('hex')


def bip32_private_key(sequence, k, chain):
    for i in sequence:
        k, chain = CKD(k, chain, i)
    return SecretToASecret(k, True)

###################################### test_crypto ##############################

def test_crypto():

    G = generator_secp256k1
    _r  = G.order()
    pvk = ecdsa.util.randrange( pow(2,256) ) %_r

    Pub = pvk*G
    pubkey_c = point_to_ser(Pub,True)
    pubkey_u = point_to_ser(Pub,False)
    addr_c = public_key_to_bc_address(pubkey_c)
    addr_u = public_key_to_bc_address(pubkey_u)

    print "Private key            ", '%064x'%pvk
    print "Compressed public key  ", pubkey_c.encode('hex')
    print "Uncompressed public key", pubkey_u.encode('hex')

    message = "Chancellor on brink of second bailout for banks"
    enc = EC_KEY.encrypt_message(message,pubkey_c)
    eck = EC_KEY(number_to_string(pvk,_r))
    dec = eck.decrypt_message(enc)
    print "decrypted", dec

    signature = eck.sign_message(message, True, addr_c)
    print signature
    EC_KEY.verify_message(addr_c, signature, message)
    
###################################### KEYTREE ##############################

import sys
import binascii
import getpass

"""
 Option to specify seed, extended key and chain in command line is possible,
 but is discouraged because on most OS commands are stored in a history file.
 To do it put the noprompt option at the begining.
 ./kt.py --noprompt -s "this is a password" --chain "(0-1)'/(6-8)'" -trav levelorder
 ./kt.py -np --extkey "xprv9uHRZZhk6KAJC1avXpDAp4MDc3sQKNxDiPvvkX8Br5ngLNv1TxvUxt4cV1rGL5hj6KCesnDYUhd7oWgT11eZG7XnxHrnYeSvkzY7d2bhkJ7" -c "(0-1)'/8"
"""

noInputEcho = False
NO_INPUT_ECHO = "-noecho"
NO_INPUT_ECHO_SHORT = "ne"
TESTNET = "-testnet"
TESTNET_SHORT = "tn"
HASH_SEED = "-hashseed"
HASH_SEED_SHORT = "hs"
NO_PROMPT = "-noprompt"
NO_PROMPT_SHORT = "np"
cmdName = "./kt"
HELP = "-help"
SEED_FORMAT = "seed_format"
SEED_VALUE = "seed_value"
EXTENDEDKEY_VALUE = "extkey_value"
CHAIN_VALUE = "chain_value"
SEED = "-seed"
SEED_HEX = "-seedhex"
EXTENDEDKEY = "-extkey"
CHAIN = "-chain"
TREE_TRAVERSAL_OPTION = "-traverse"
TREE_TRAVERSAL_TYPE_PREORDER = "preorder"
TREE_TRAVERSAL_TYPE_POSTORDER = "postorder"
TREE_TRAVERSAL_TYPE_LEVELORDER = "levelorder"
OUTPUT_ENTIRE_CHAIN_OPTION = "-all"
VERBOSE_OPTION = "-verbose"
SEED_SHORT = "s"
SEED_HEX_SHORT = "sh"
EXTENDEDKEY_SHORT = "ek"
CHAIN_SHORT = "c"
TREE_TRAVERSAL_OPTION_SHORT = "trav"
TREE_TRAVERSAL_TYPE_PREORDER_SHORT = "pre"
TREE_TRAVERSAL_TYPE_POSTORDER_SHORT = "post"
TREE_TRAVERSAL_TYPE_LEVELORDER_SHORT = "lev"
OUTPUT_ENTIRE_CHAIN_OPTION_SHORT = "a"
VERBOSE_OPTION_SHORT = "v"

class StringType:
    HEX = 1
    BASE58 = 2
    ASCII = 3 

class TreeTraversal:
    PREORDER = 1
    POSTORDER = 2
    LEVELORDER = 3 

DEFAULTTREETRAVERSALTYPE = TreeTraversal.PREORDER



class KeyTreeUtil(object):
    NODE_IDX_M_FLAG = sys.maxint
    MASTER_NODE_LOWERCASE_M = "m"
    LEAD_CHAIN_PATH = "___"

    @staticmethod
    def sha256Rounds(data, rounds):
        for i in xrange(rounds):
            data = sha256(data)
        return data

    @staticmethod
    def toPrime(i):
        return i + BIP32_PRIME  

    @staticmethod
    def removePrime(i):
        return 0x7fffffff & i

    @staticmethod
    def isPrime(i):
        return BIP32_PRIME & i  

    @staticmethod
    def iToString(i):    
        if KeyTreeUtil.isPrime(i):
            return str(KeyTreeUtil.removePrime(i))+"'"
        else:
            return str(i)  

    @staticmethod
    def parseRange(node, isPrivate):
        #node must be like (123-9345)
        minMaxString = node[1:-1]
        minMaxPair = minMaxString.split('-')
        if len(minMaxPair) != 2:
            raise ValueError('Invalid arguments.')
        min = int(minMaxPair[0])
        max = int(minMaxPair[1])
        if isPrivate:
            return [True, [min, max]]
        else:
            return [False, [min, max]]

    @staticmethod
    def parseChainString(sequence):
        treeChains = []
        splitChain = sequence.split('/')

        treeChains.append([True, [KeyTreeUtil.NODE_IDX_M_FLAG, KeyTreeUtil.NODE_IDX_M_FLAG]])

        for node in splitChain:
            if node[-1] == "'":
                node = node[:-1]
                if node[0] == '(' and node[-1] == ')':
                    treeChains.append(KeyTreeUtil.parseRange(node, True))
                else:
                    num = int(node)
                    treeChains.append([True, [num, num]])
            else:
                if node[0] == '(' and node[-1] == ')':
                    treeChains.append(KeyTreeUtil.parseRange(node, False))
                else:
                    num = int(node)
                    treeChains.append([False, [num, num]])

        return treeChains

    @staticmethod
    def powMod(x, y, z):
        """Calculate (x ** y) % z efficiently."""
        number = 1
        while y:
            if y & 1:
                number = number * x % z
            y >>= 1
            x = x * x % z
        return number

    @staticmethod
    def compressedPubKeyToUncompressedPubKey(compressedPubKey):
        compressedPubKey = compressedPubKey.encode('hex')

        p = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
        y_parity = int(compressedPubKey[:2]) - 2

        x = int(compressedPubKey[2:], 16)
        a = (KeyTreeUtil.powMod(x, 3, p) + 7) % p
        y = KeyTreeUtil.powMod(a, (p+1)//4, p)
        if y % 2 != y_parity:
            y = -y % p

        ppp = KeyTreeUtil.powMod(x, 3, p)
        uncompressedPubKey = '{:064x}{:064x}'.format(x, y)
        return uncompressedPubKey.decode('hex')



class KeyNode(object):
    priv_version = 0x0488ADE4
    pub_version = 0x0488B21E
    addr_type = 0

    def __init__(self, key = None, chain_code = None, extkey = None, child_num = 0, parent_fp = 0, depth = 0):
        self.version = None
        self.depth = None
        self.parent_fp = None
        self.child_num = None
        self.chain_code = None
        self.key = None
        self.valid = False
        self.pubkey = None
        self.pubkey_compressed = None

        if key and chain_code:
            self.key = key
            self.chain_code = chain_code
            self.child_num = child_num
            self.parent_fp = format(parent_fp, '#010x')[2:].decode('hex')
            self.depth = depth
            self.version = KeyNode.priv_version
            if self.key:
                if len(self.key) == 32:
                    self.key = '\00'+self.key
                elif len(self.key) != 33:
                    raise ValueError('Invalid key.')

                K0, K0_compressed = get_pubkeys_from_secret(self.key[1:])
                self.pubkey = K0
                self.pubkey_compressed = K0_compressed
            
            self.valid = True
        elif extkey:
            self.parseExtKey(extkey)

    def parseExtKey(self, extKey):
        if len(extKey) != 78:
            raise ValueError("Invalid extended key length.")

        self.version = extKey[0:4]
        self.depth = extKey[4]
        self.parent_fp = extKey[5:9]
        self.child_num = extKey[9:13]
        self.chain_code = extKey[13:45]
        self.key = extKey[45:78]

        self.version = int(self.version.encode('hex'), 16)
        self.depth = int(self.depth.encode('hex'), 16)
        self.child_num = int(self.child_num.encode('hex'), 16)

        if self.isPrivate():
            if self.version != KeyNode.priv_version:
                raise ValueError("Invalid extended key version.")            

            K0, K0_compressed = get_pubkeys_from_secret(self.key[1:])
            self.pubkey = K0
            self.pubkey_compressed = K0_compressed
        else:
            if self.version != KeyNode.pub_version:
                raise ValueError("Invalid extended key version.")            

            self.pubkey_compressed = self.key
            self.pubkey = KeyTreeUtil.compressedPubKeyToUncompressedPubKey(self.key)

        self.valid = True

    def getVersion(self):
        return self.version

    def getDepth(self):
        return self.depth

    def getParentFingerPrint(self):
        return self.parent_fp

    def getChildNum(self):
        return self.child_num

    def getChainCodeBytes(self):
        return self.chain_code

    def getKeyBytes(self):
        return self.key

    def getPubKeyBytes(self, compressed):
        if compressed:
            return self.pubkey_compressed
        else:
            return self.pubkey

    def isPrivate(self):
        return len(self.key) == 33 and int(self.key[0].encode('hex'), 16) == 0x00

    def getFingerPrint(self):
        return hash_160(self.pubkey_compressed)[:4]

    def getPublic(self):
        if not self.valid:
            raise Exception('Keychain is invalid.')

        pub = KeyNode()
        pub.valid = self.valid
        pub.version = KeyNode.pub_version
        pub.depth = self.depth
        pub.parent_fp = self.parent_fp
        pub.child_num = self.child_num
        pub.chain_code = self.chain_code
        pub.pubkey = self.pubkey
        pub.pubkey_compressed = self.pubkey_compressed
        pub.key = self.pubkey_compressed
        return pub

    def getChild(self, i):
        if not self.valid:
            raise Exception('Keychain is invalid.')

        if not self.isPrivate() and KeyTreeUtil.isPrime(i):
            raise Exception('Cannot do private key derivation on public key.')

        child = KeyNode()
        child.valid = False
        child.version = self.version
        child.depth = self.depth + 1
        child.parent_fp = self.getFingerPrint()
        child.child_num = i

        if self.isPrivate():
            child.key, child.chain_code = CKD(self.key[1:], self.chain_code, i)
            # pad with 0's to make it 33 bytes
            zeroPadding = '\00'*(33 - len(child.key))
            child.key = zeroPadding + child.key
            child.pubkey, child.pubkey_compressed = get_pubkeys_from_secret(child.key[1:])
        else:
            child.pubkey, child.pubkey_compressed, child.chain_code = CKD_prime(self.pubkey, self.chain_code, i)
            child.key = child.pubkey_compressed

        child.valid = True
        return child

    def getPrivKey(self, compressed):
        return SecretToASecret(self.key[1:], compressed, KeyNode.addr_type)

    def getPubKey(self, compressed):
        if compressed:
            return self.pubkey_compressed.encode('hex')
        else:
            return ('\04' + self.pubkey).encode('hex')

    def getAddress(self, compressed):
        if compressed:
            return hash_160_to_bc_address(hash_160(self.pubkey_compressed), KeyNode.addr_type)
        else:
            return hash_160_to_bc_address(hash_160('\04' + self.pubkey), KeyNode.addr_type)

    def getExtKey(self):
        depthBytes = format(self.depth, '#04x')[2:].decode('hex')
        childNumBytes = format(self.child_num, '#010x')[2:].decode('hex')
        versionBytes = format(self.version, '#010x')[2:].decode('hex')
        extkey = versionBytes+depthBytes+self.parent_fp+childNumBytes+self.chain_code+self.key
        return EncodeBase58Check(extkey)

    @staticmethod
    def setTestNet(enabled):
        if enabled:
            KeyNode.priv_version = 0x04358394
            KeyNode.pub_version = 0x043587CF
            KeyNode.addr_type = 111
        else:
            KeyNode.priv_version = 0x0488ADE4
            KeyNode.pub_version = 0x0488B21E
            KeyNode.addr_type = 0


def testVector1():
    optionsDict = {OUTPUT_ENTIRE_CHAIN_OPTION:True, VERBOSE_OPTION:False}
    #optionsDict[VERBOSE_OPTION] = True
    outputExtKeysFromSeed("000102030405060708090a0b0c0d0e0f", "0'/1/2'/2/1000000000", StringType.HEX, 0, optionsDict)

def testVector2():
    optionsDict = {OUTPUT_ENTIRE_CHAIN_OPTION:True, VERBOSE_OPTION:False}
    #optionsDict[VERBOSE_OPTION] = True
    seed = "fffcf9f6f3f0edeae7e4e1dedbd8d5d2cfccc9c6c3c0bdbab7b4b1aeaba8a5a29f9c999693908d8a8784817e7b7875726f6c696663605d5a5754514e4b484542"
    outputExtKeysFromSeed(seed,"0/2147483647'/1/2147483646'/2", StringType.HEX, 0, optionsDict)
    
def parse_arguments(argv):
    argsDict = {}
    it = 0
    while it < len(argv):
        arg = argv[it]
        if arg[0] != '-':
            raise ValueError("Invalid arguments.")

        arg = arg[1:]
        if arg == HELP:
            argsDict[HELP] = HELP
            break
        elif arg == SEED or arg == SEED_SHORT:
            argsDict[SEED_FORMAT] = "" #assumes ascii
            argsDict[SEED] = "Y"
            if getOptionValue(argsDict.get(NO_PROMPT)):
                it += 1
                argsDict[SEED_VALUE] = argv[it]
        elif arg == SEED_HEX or arg == SEED_HEX_SHORT:
            argsDict[SEED_FORMAT] = "hex"
            argsDict[SEED] = "Y"
            if getOptionValue(argsDict.get(NO_PROMPT)):
                it += 1
                argsDict[SEED_VALUE] = argv[it]
        elif arg == EXTENDEDKEY or arg == EXTENDEDKEY_SHORT:
            argsDict[EXTENDEDKEY] = "Y"
            
            if getOptionValue(argsDict.get(NO_PROMPT)):
                it += 1
                argsDict[EXTENDEDKEY_VALUE] = argv[it]
        elif arg == CHAIN or arg == CHAIN_SHORT:
            if getOptionValue(argsDict.get(NO_PROMPT)):
                it += 1
                argsDict[CHAIN_VALUE] = argv[it]
        elif arg == TREE_TRAVERSAL_OPTION or arg == TREE_TRAVERSAL_OPTION_SHORT:
            it += 1
            argsDict[TREE_TRAVERSAL_OPTION] = argv[it]
            argsDict[OUTPUT_ENTIRE_CHAIN_OPTION] = "Y"
        elif arg == OUTPUT_ENTIRE_CHAIN_OPTION or arg == OUTPUT_ENTIRE_CHAIN_OPTION_SHORT:
            argsDict[OUTPUT_ENTIRE_CHAIN_OPTION] = "Y"
        elif arg == VERBOSE_OPTION or arg == VERBOSE_OPTION_SHORT:
            argsDict[VERBOSE_OPTION] = "Y"
        elif arg == NO_INPUT_ECHO or arg == NO_INPUT_ECHO_SHORT:
            global noInputEcho
            noInputEcho = True
        elif arg == TESTNET or arg == TESTNET_SHORT:
            argsDict[TESTNET] = "Y"
        elif arg == HASH_SEED or arg == HASH_SEED_SHORT:
            argsDict[HASH_SEED] = "Y"
            
            if getOptionValue(argsDict.get(NO_PROMPT)):
                it += 1
                argsDict[HASH_SEED] = argv[it]
        elif arg == NO_PROMPT or arg == NO_PROMPT_SHORT:
            argsDict[NO_PROMPT] = "Y"
        else:
            raise ValueError("Invalid arguments.")

        it += 1

    # default to seed if no option provided
    if argsDict.get(EXTENDEDKEY) == None and argsDict.get(SEED) == None:
        argsDict[SEED] = "Y"
    
    return argsDict

def outputExamples():
    outputString("Extended Keys can be in hex or base58. Seed can be in ASCII or hex. Examples below.")
    outputString("")
    
    outputString("To use KeyTree simply do the following:")
    outputString(cmdName)
    outputString("Enter Seed:")
    outputString("correct horse battery staple")
    outputString("Enter Chain:")
    outputString("0'/0")
    outputString("")
    
    outputString("Use the hex option to enter the seed in hex:")
    outputString(cmdName+" --seed.hex")
    outputString("Enter Seed in Hex:")
    outputString("000102030405060708090a0b0c0d0e0f")
    outputString("Enter Chain:")
    outputString("0'/0")
    outputString("")
    
    outputString("Use the extended key option to enter the extended key in lieu of the seed:")
    outputString(cmdName+" --extkey")
    outputString(cmdName+" -ek")
    outputString("")
    
    outputString("It is also possible to print multiple chain paths together:")
    outputString(cmdName)
    outputString("Enter Extended Key:")
    outputString("xprv9uHRZZhk6KAJC1avXpDAp4MDc3sQKNxDiPvvkX8Br5ngLNv1TxvUxt4cV1rGL5hj6KCesnDYUhd7oWgT11eZG7XnxHrnYeSvkzY7d2bhkJ7")
    outputString("Enter Chain:")
    outputString("0'/(3-6)'/(1-2)/8")
    outputString("")
    
    outputString("To output all the node data on the chain, use the all option:")
    outputString(cmdName+" --all")
    outputString(cmdName+" -a")
    outputString("")
    
    outputString("It is also possible to output the nodes in a different order:")
    outputString(cmdName+" --traverse levelorder")
    outputString(cmdName+" -trav postorder")
    outputString(cmdName+" -trav preorder")
    outputString("")
    
    outputString("For more information on the node use the verbose option:")
    outputString(cmdName+" --verbose")
    outputString(cmdName+" -v")

def getTreeTraversalOption(treeTraversalOption):
    if treeTraversalOption == TREE_TRAVERSAL_TYPE_LEVELORDER or treeTraversalOption == TREE_TRAVERSAL_TYPE_LEVELORDER_SHORT:
        return TreeTraversal.LEVELORDER
    elif treeTraversalOption == TREE_TRAVERSAL_TYPE_POSTORDER or treeTraversalOption == TREE_TRAVERSAL_TYPE_POSTORDER_SHORT:
        return TreeTraversal.POSTORDER
    elif treeTraversalOption == TREE_TRAVERSAL_TYPE_PREORDER or treeTraversalOption == TREE_TRAVERSAL_TYPE_PREORDER:
        return TreeTraversal.PREORDER
    else:
        return DEFAULTTREETRAVERSALTYPE

def getOptionValue(option):
    if option == "Y": return True
    else: return False

def get_input(pretext):
    if noInputEcho:
        return getpass.getpass(pretext+'\n')
    else:   
        return raw_input(pretext+'\n')

def enter_prompt(argsDict):
    if argsDict.get(HELP) == HELP:
        outputExamples()
    else:
        optionsDict = {}
        optionsDict[TESTNET] = getOptionValue(argsDict.get(TESTNET))
        optionsDict[OUTPUT_ENTIRE_CHAIN_OPTION] = getOptionValue(argsDict.get(OUTPUT_ENTIRE_CHAIN_OPTION))
        optionsDict[VERBOSE_OPTION] = getOptionValue(argsDict.get(VERBOSE_OPTION))
        traverseType = getTreeTraversalOption(argsDict.get(TREE_TRAVERSAL_OPTION))

        if getOptionValue(argsDict.get(SEED)):
            seed = None
            seed_format = None
            if (argsDict.get(SEED_FORMAT) == "hex"):
                seed_format = StringType.HEX
                seed = get_input("Enter Seed in Hex:")
                try: int(seed, 16)
                except ValueError: raise ValueError("Invalid hex string \"" + seed + "\"")
            else:
                seed_format = StringType.ASCII
                seed = get_input("Enter Seed:")
            
            chain = get_input("Enter Chain:")
            
            roundsToHash = 0
            if getOptionValue(argsDict.get(HASH_SEED)):
                roundsToHashStr = get_input("Enter number of rounds of Sha256 hash:")
                if roundsToHashStr:
                    roundsToHash = int(roundsToHashStr)
            
            outputExtKeysFromSeed(seed, chain, seed_format, roundsToHash, optionsDict, traverseType)
            
        elif getOptionValue(argsDict.get(EXTENDEDKEY)):
            extkey = get_input("Enter Extended Key:")
            chain = get_input("Enter Chain:")
            
            if chain != "":
                outputExtKeysFromExtKey(extkey, chain, optionsDict, traverseType)
            else:
                outputKeyAddressofExtKey(extkey, optionsDict)
    
    return 0

def handle_arguments(argsDict):
    outputString("Arguments:")
    for key in argsDict:
        outputString("\tkey: " + key + " value: " + argsDict[key])
    outputString("")

    if argsDict.get(HELP) == HELP:
        outputExamples()
        return 0
    else:
        optionsDict = {}
        optionsDict[TESTNET] = getOptionValue(argsDict.get(TESTNET))
        optionsDict[OUTPUT_ENTIRE_CHAIN_OPTION] = getOptionValue(argsDict.get(OUTPUT_ENTIRE_CHAIN_OPTION))
        optionsDict[VERBOSE_OPTION] = getOptionValue(argsDict.get(VERBOSE_OPTION))

        if argsDict.get(SEED_VALUE) != None and argsDict.get(CHAIN_VALUE) != None:
            seed = argsDict.get(SEED_VALUE)
            chain = argsDict.get(CHAIN_VALUE)
            
            seed_format = None
            if argsDict.get(SEED_FORMAT) == "hex":
                seed_format = StringType.HEX
            else:
                seed_format = StringType.ASCII
            
            roundsToHashStr = argsDict.get(HASH_SEED)
            roundsToHash = 0
            if roundsToHashStr:
                roundsToHash = int(roundsToHashStr)
            
            traverseType = getTreeTraversalOption(argsDict.get(TREE_TRAVERSAL_OPTION))
            optionsDict[HASH_SEED] = getOptionValue(argsDict.get(HASH_SEED))
            outputExtKeysFromSeed(seed, chain, seed_format, roundsToHash, optionsDict, traverseType)
        elif argsDict.get(EXTENDEDKEY_VALUE) != None and argsDict.get(CHAIN_VALUE) != None:
            extkey = argsDict.get(EXTENDEDKEY_VALUE)
            chain = argsDict.get(CHAIN_VALUE)
            
            traverseType = getTreeTraversalOption(argsDict.get(TREE_TRAVERSAL_OPTION))
            outputExtKeysFromExtKey(extkey, chain, optionsDict, traverseType)
        elif argsDict.get(EXTENDEDKEY) != None:
            extkey = argsDict.get(EXTENDEDKEY_VALUE)
            outputKeyAddressofExtKey(extkey, optionsDict)
        else:
            raise ValueError("Invalid arguments.")

    return 0

def outputString(string):
    print string

def visit(keyNode, chainName, isLeafNode, optionsDict):
    if not isLeafNode and not optionsDict.get(OUTPUT_ENTIRE_CHAIN_OPTION):
        return

    outputString("* [Chain " + chainName + "]")
    if keyNode.isPrivate():
        keyNodePub = keyNode.getPublic()
        outputString("  * ext pub:  " + keyNodePub.getExtKey())
        outputString("  * ext prv:  " + keyNode.getExtKey())
        if optionsDict.get(VERBOSE_OPTION) == False:
            outputString("  * priv key: " + keyNode.getPrivKey(True))
            outputString("  * address:  " + keyNode.getAddress(True))
        else:
            outputString("  * uncompressed priv key: " + keyNode.getPrivKey(False))
            outputString("  * uncompressed pub key:  " + keyNode.getPubKey(False))
            outputString("  * uncompressed address:  " + keyNode.getAddress(False))
            outputString("  * compressed priv key: " + keyNode.getPrivKey(True))
            outputString("  * compressed pub key:  " + keyNode.getPubKey(True))
            outputString("  * compressed address:  " + keyNode.getAddress(True))
    else:
        outputString("  * ext pub:  " + keyNode.getExtKey())
        if optionsDict[VERBOSE_OPTION] == False:
            outputString("  * address:  " + keyNode.getAddress(True))
        else:
            outputString("  * uncompressed pub key:  " + keyNode.getPubKey(False))
            outputString("  * uncompressed address:  " + keyNode.getAddress(False))
            outputString("  * compressed pub key:  " + keyNode.getPubKey(True))
            outputString("  * compressed address:  " + keyNode.getAddress(True))

def traversePreorder(keyNode, treeChains, chainName, optionsDict):
    if treeChains:
        isPrivateNPathRange = treeChains.pop(0)
        isPrivate = isPrivateNPathRange[0]
        min = isPrivateNPathRange[1][0]
        max = isPrivateNPathRange[1][1]
        isLeafNode = False
        if not treeChains: isLeafNode = True
        if min == KeyTreeUtil.NODE_IDX_M_FLAG and max == KeyTreeUtil.NODE_IDX_M_FLAG:
            visit(keyNode, chainName, isLeafNode, optionsDict)
            traversePreorder(keyNode, treeChains[:], chainName, optionsDict)
        else:
            for i in range(min, max+1):
                childChainName = chainName + "/" + str(i) + "'" if isPrivate else chainName + "/" + str(i)
                if isPrivate:
                    childNode = keyNode.getChild(KeyTreeUtil.toPrime(i))
                else:
                    childNode = keyNode.getChild(i)

                visit(childNode, childChainName, isLeafNode, optionsDict)
                traversePreorder(childNode, treeChains[:], childChainName, optionsDict)

def traversePostorder(keyNode, treeChains, chainName, optionsDict):
    if treeChains:
        isPrivateNPathRange = treeChains.pop(0)
        isPrivate = isPrivateNPathRange[0]
        min = isPrivateNPathRange[1][0]
        max = isPrivateNPathRange[1][1]
        isLeafNode = False
        if not treeChains: isLeafNode = True
        if min == KeyTreeUtil.NODE_IDX_M_FLAG and max == KeyTreeUtil.NODE_IDX_M_FLAG:
            traversePostorder(keyNode, treeChains[:], chainName, optionsDict)
            visit(keyNode, chainName, isLeafNode, optionsDict)
        else:
            for i in range(min, max+1):
                if isPrivate: i = KeyTreeUtil.toPrime(i)
                childChainName = chainName + "/" + KeyTreeUtil.iToString(i)
                childNode = keyNode.getChild(i)

                traversePostorder(childNode, treeChains[:], childChainName, optionsDict)
                visit(childNode, childChainName, isLeafNode, optionsDict)

def traverseLevelorder(keyNode, treeChains, chainName, level, keyNodeDeq, levelNChainDeq, optionsDict):
    isLeafNode = False
    if level < len(treeChains):
        isPrivateNPathRange = treeChains[level]
        isPrivate = isPrivateNPathRange[0]
        min = isPrivateNPathRange[1][0]
        max = isPrivateNPathRange[1][1]
        level += 1
        for i in range(min, max+1):
            if isPrivate: i = KeyTreeUtil.toPrime(i)
            childChainName = chainName + "/" + KeyTreeUtil.iToString(i)
            childNode = keyNode.getChild(i)
            keyNodeDeq.append(childNode)            
            levelNChainDeq.append([level, childChainName])            
    else:
        isLeafNode = True

    visit(keyNode, chainName, isLeafNode, optionsDict)

    if keyNodeDeq:
        pair = levelNChainDeq.pop(0)
        level = pair[0] 
        cc = pair[1] 
        node = keyNodeDeq.pop(0)
        traverseLevelorder(node, treeChains, cc, level, keyNodeDeq, levelNChainDeq, optionsDict)

def outputExtraKeyNodeData(keyNode):
    outputString("  * depth:              " + str(keyNode.getDepth()))
    outputString("  * child number:       " + KeyTreeUtil.iToString(keyNode.getChildNum()))
    outputString("  * parent fingerprint: " + keyNode.getParentFingerPrint().encode('hex'))
    outputString("  * fingerprint:        " + keyNode.getFingerPrint().encode('hex'))

def outputExtKeysFromSeed(seed, chainStr, seedStringFormat, roundsToHash, optionsDict, traverseType = DEFAULTTREETRAVERSALTYPE):
    seedHexStr = None
    if seedStringFormat == StringType.ASCII:
        seedHexStr = binascii.hexlify(seed)
    elif seedStringFormat == StringType.HEX:
        try: int(seed, 16)
        except ValueError: raise ValueError("Invalid hex string \"" + seed + "\"")
        seedHexStr = seed
    else:
        raise ValueError("Invalid seed string format.")

    if roundsToHash > 0:
        seedHexStr = KeyTreeUtil.sha256Rounds(seedHexStr.decode('hex') , roundsToHash).encode('hex')

    if optionsDict.get(TESTNET) == None or optionsDict.get(TESTNET) == False:
        KeyNode.setTestNet(False)
    else:
        KeyNode.setTestNet(True)

    master_secret, master_chain, master_public_key, master_public_key_compressed = bip32_init(seedHexStr)
    k = master_secret
    c = master_chain

    keyNodeSeed = KeyNode(key = k, chain_code = c)

    treeChains = KeyTreeUtil.parseChainString(chainStr)
    outputString("Master (hex): " + seedHexStr)

    if traverseType == TreeTraversal.POSTORDER:
        traversePostorder(keyNodeSeed, treeChains, KeyTreeUtil.MASTER_NODE_LOWERCASE_M, optionsDict)
    elif traverseType == TreeTraversal.LEVELORDER:
        treeChains.pop(0)
        traverseLevelorder(keyNodeSeed, treeChains, KeyTreeUtil.MASTER_NODE_LOWERCASE_M, 0, [], [], optionsDict)
    else:        
        traversePreorder(keyNodeSeed, treeChains, KeyTreeUtil.MASTER_NODE_LOWERCASE_M, optionsDict)

def outputExtKeysFromExtKey(extKey, chainStr, optionsDict, traverseType = DEFAULTTREETRAVERSALTYPE):
    if optionsDict.get(TESTNET) == None or optionsDict.get(TESTNET) == False:
        KeyNode.setTestNet(False)
    else:
        KeyNode.setTestNet(True)

    keyNode = None
    try:
        int(extKey, 16)
        keyNode = KeyNode(extkey = extKey.decode('hex'))
    except ValueError:
        extKeyBytes = DecodeBase58Check(extKey)
        if not extKeyBytes:
            raise ValueError('Invalid extended key.')
        keyNode = KeyNode(extkey = extKeyBytes)

    treeChains = KeyTreeUtil.parseChainString(chainStr)

    if optionsDict.get(VERBOSE_OPTION): outputExtraKeyNodeData(keyNode)

    if traverseType == TreeTraversal.POSTORDER:
        traversePostorder(keyNode, treeChains, KeyTreeUtil.LEAD_CHAIN_PATH, optionsDict)
    elif traverseType == TreeTraversal.LEVELORDER:
        treeChains.pop(0)
        traverseLevelorder(keyNode, treeChains, KeyTreeUtil.LEAD_CHAIN_PATH, 0, [], [], optionsDict)
    else:        
        traversePreorder(keyNode, treeChains, KeyTreeUtil.LEAD_CHAIN_PATH, optionsDict)

def outputKeyAddressofExtKey(extKey, optionsDict):
    if optionsDict.get(TESTNET) == None or optionsDict.get(TESTNET) == False:
        KeyNode.setTestNet(False)
    else:
        KeyNode.setTestNet(True)

    extKeyBytes = DecodeBase58Check(extKey)
    if not extKeyBytes:
        raise ValueError('Invalid extended key.')

    keyNode = KeyNode(extkey = DecodeBase58Check(extKey))
    if optionsDict.get(VERBOSE_OPTION): outputExtraKeyNodeData(keyNode)
    visit(keyNode, KeyTreeUtil.LEAD_CHAIN_PATH, True, optionsDict)
    outputString("")

def main():
    argv = sys.argv[1:]
    argsDict = parse_arguments(argv)

    if getOptionValue(argsDict.get(NO_PROMPT)):
        return handle_arguments(argsDict)
    else:
        return enter_prompt(argsDict)

if __name__ == '__main__':
    #test_crypto()
    #testVector1()
    #testVector2()
    main()
