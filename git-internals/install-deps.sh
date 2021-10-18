sudo apt-get install -y cabal-install
cabal update
git clone git@github.com:owickstrom/pandoc-include-code.git
cd pandoc-include-code
cabal configure
cabal install
cd .. && rm -rf ./pandoc-include-code
ln -s ~/.cabal/bin/pandoc-include-code ~/.local/bin/pandoc-include-code
 