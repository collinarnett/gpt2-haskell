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
        ...
      }: let
        ghc = "ghc984";
      in {
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          config.cudaSupport = false;
          overlays = [
            (final: prev: {
              libtorch-bin = prev.libtorch-bin.overrideAttrs {dontStrip = true;};
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
