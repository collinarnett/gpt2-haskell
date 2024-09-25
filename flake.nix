{
  nixConfig = {
    extra-substituters = [
      "https://hasktorch.cachix.org"
    ];
    extra-trusted-public-keys = [
      "hasktorch.cachix.org-1:wLjNS6HuFVpmzbmv01lxwjdCOtWRD8pQVR3Zr/wVoQc="
    ];
  };

  inputs = {
    hasktorch.url = "github:hasktorch/hasktorch";
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.follows = "hasktorch/nixpkgs";
  };
  outputs = inputs @ {
    self,
    nixpkgs,
    flake-parts,
    hasktorch,
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      perSystem = {
        system,
        pkgs,
        lib,
        ...
      }: let
        ghc = "ghc965";
      in {
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          config.cudaSupport = false;
          overlays = [
            (final: prev: {
              libtorch-bin = prev.libtorch-bin.overrideAttrs {dontStrip = true;};
              haskell =
                prev.haskell
                // {
                  packages =
                    prev.haskell.packages
                    // {
                      ${ghc} = prev.haskell.packages.${ghc}.extend (fHaskellPackages: pHaskellPackages: {
                        "tiktoken" =
                          pHaskellPackages.callPackage
                          ({
                            mkDerivation,
                            base,
                            base64_1_0,
                            bytestring,
                            containers,
                            deepseq,
                            filepath,
                            megaparsec,
                            pcre-light,
                            quickcheck-instances,
                            raw-strings-qq,
                            tasty,
                            tasty-bench,
                            tasty-quickcheck,
                            tasty-silver,
                            text,
                            unordered-containers,
                          }:
                            mkDerivation {
                              pname = "tiktoken";
                              version = "1.0.3";
                              sha256 = "0hy3y9rdgjirk8ji7458qnc7h9d2b6yipfri25qkay96kq91kmj6";
                              enableSeparateDataOutput = true;
                              libraryHaskellDepends = [
                                base
                                base64_1_0
                                bytestring
                                containers
                                deepseq
                                filepath
                                megaparsec
                                pcre-light
                                raw-strings-qq
                                text
                                unordered-containers
                              ];
                              testHaskellDepends = [
                                base
                                bytestring
                                quickcheck-instances
                                tasty
                                tasty-quickcheck
                                tasty-silver
                                text
                              ];
                              benchmarkHaskellDepends = [
                                base
                                bytestring
                                deepseq
                                filepath
                                tasty-bench
                              ];
                              description = "Haskell implementation of tiktoken";
                              license = lib.licenses.bsd3;
                            }) {};
                      });
                    };
                };
            })
            hasktorch.overlays.default
          ];
        };
        devShells.default = pkgs.haskell.packages.${ghc}.shellFor {
          packages = let
            gpt2-haskell = ps: (ps.callCabal2nix "gpt2-haskell" ./gpt2-haskell {});
          in
            ps: [
              (gpt2-haskell ps)
              (ps.callCabal2nix "examples" ./examples {gpt2-haskell = gpt2-haskell ps;})
            ];
          nativeBuildInputs = with pkgs; [
            cabal-install
            haskellPackages.cabal-fmt
            haskell-language-server
            ormolu
            stylish-haskell
          ];
          withHoogle = true;
        };
      };
    };
}
