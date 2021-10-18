sudo apt-get install -y cabal-install
cabal update
wget https://github.com/owickstrom/pandoc-include-code/archive/refs/heads/master.zip
unzip master.zip
rm -f master.zip
cd pandoc-include-code-master
cabal configure
cabal install
cd .. && rm -rf ./pandoc-include-code-master
export PATH=$PATH:~/.cabal/bin
rm -rf dist
 