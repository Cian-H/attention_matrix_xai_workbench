{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = [
    pkgs.git
    pkgs.marimo
    pkgs.ruff
  ];
  env = {
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.libGL
      pkgs.libGLU
      pkgs.wlroots
      pkgs.ncurses5
    ]
    + ":/run/opengl-driver/lib"
    + ":/usr/lib/x86_64-linux-gnu"
    + ":/usr/lib/wsl/lib";
    NIX_LD = lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
    CUDA_PATH = "${pkgs.cudatoolkit}";
    EXTRA_LDFLAGS = "-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib";
    EXTRA_CCFLAGS = "-I/usr/include";
  };

  languages.python = {
    enable = true;
    package = pkgs.python314;
    lsp.package = pkgs.ty;
    uv.enable = true;
    uv.sync.enable = true;
  };
}
