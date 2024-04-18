wget http://www.cs.toronto.edu/~complingweb/data/TORGO/F.tar.bz2 -P Torgo/
wget http://www.cs.toronto.edu/~complingweb/data/TORGO/FC.tar.bz2 -P Torgo/
wget http://www.cs.toronto.edu/~complingweb/data/TORGO/M.tar.bz2 -P Torgo/
wget http://www.cs.toronto.edu/~complingweb/data/TORGO/MC.tar.bz2 -P Torgo/

tar -xvf Torgo/F.tar.bz2 -C Torgo/dysarthria
tar -xvf Torgo/FC.tar.bz2 -C Torgo/non_dysarthria
tar -xvf Torgo/M.tar.bz2 -C Torgo/dysarthria
tar -xvf Torgo/MC.tar.bz2 -C Torgo/non_dysarthria