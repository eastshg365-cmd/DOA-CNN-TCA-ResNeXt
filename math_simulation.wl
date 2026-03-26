(* ::Package:: *)

(* ============================================================
   DOA-CNN-TCA-ResNeXt: Mathematica \:6570\:5b66\:6f14\:7b97
   EIE4127 FYP | PolyU
   \:5bf9\:5e94\:8bba\:6587: IEEE MLSP 2020  DOI: 10.1109/MLSP49062.2020.9231787
   ============================================================ *)

(* ============================================================
   \:7b2c1\:8282: TCA \:9635\:5217\:51e0\:4f55
   ============================================================ *)

(* TCA \:53c2\:6570 M=5, Narr=6, gcd(5,6)=1 \:4e92\:8d28\:9a8c\:8bc1 *)
(* \:6ce8\:610f: \:4e0d\:80fd\:7528 N \:4f5c\:53d8\:91cf\:540d\:ff0cN \:662f Mathematica \:5185\:5efa\:6570\:503c\:51fd\:6570\:ff08\:53d7\:4fdd\:62a4\:7b26\:53f7\:ff09*)
M = 5; Narr = 6;
Print["<gcd(M,Narr) = >", GCD[M, Narr], "<  (\:5fc5\:987b\:4e3a1\:ff0c\:9a8c\:8bc1\:4e92\:8d28)>"]

(* \:4e09\:4e2a\:5b50\:9635 *)
X1 = Table[n*M, {n, 0, Narr - 1}];
X2 = Table[m*Narr, {m, 1, Floor[M/2]}];
X3 = Table[(m + M + 1)*Narr, {m, 0, M - 2}];

(* TCA = \:5e76\:96c6 + \:6392\:5e8f\:ff0c\:5f3a\:5236\:6574\:6570\:5217\:8868\:4ee5\:907f\:514d\:540e\:7eed\:7b26\:53f7\:8fd0\:7b97 *)
TCA = N[Sort[Union[X1, X2, X3]]];

Print["<X1 = >", X1]
Print["<X2 = >", X2]
Print["<X3 = >", X3]
Print["<TCA = >", TCA]
Print["<\:4f20\:611f\:5668\:6570 P = >", Length[TCA], "<  (\:516c\:5f0f: M+Narr+Floor[M/2]-1 = >",
  M + Narr + Floor[M/2] - 1, "<)>"]
Print["<\:9635\:5217\:5b54\:5f84 = >", Max[TCA], "<d>"]

(* \:53ef\:89c6\:5316 TCA \:4f4d\:7f6e *)
tcaPlot = Graphics[{
    Table[{Disk[{TCA[[i]], 0}, 0.3]}, {i, Length[TCA]}],
    {Dashed, Line[{{0, -0.8}, {54, -0.8}}]},
    Table[Text[TCA[[i]], {TCA[[i]], -1.5}], {i, Length[TCA]}]
  },
  Frame -> True,
  FrameLabel -> {"\:4f4d\:7f6e (\[Times]d=\[Lambda]/2)", ""},
  PlotLabel -> "TCA \:4f20\:611f\:5668\:4f4d\:7f6e (M=5, N=6)",
  ImageSize -> 700, AspectRatio -> 0.2
];
tcaPlot

(* ============================================================
   \:7b2c2\:8282: \:5dee\:5206\:9635\:5217 (Difference Co-array)
   ============================================================ *)

diffArray = Sort[Union[Flatten[
    Outer[Subtract, TCA, TCA]
]]];
Print["<\:5dee\:5206\:9635\:5217\:5143\:7d20\:6570 = >", Length[diffArray]]
Print["<\:5dee\:5206\:9635\:5217\:8303\:56f4: [>", Min[diffArray], "<, >", Max[diffArray], "<]>"]

(* \:8fde\:7eed\:90e8\:5206 (lags = 0, 1, 2, ...) *)
consecutiveLags = Module[{i = 0, lags = {}},
  While[MemberQ[diffArray, N[i]],
    AppendTo[lags, i]; i++];
  lags
];
Print["<\:8fde\:7eed\:6b63\:8fdf\:540e\:6570 = >", Length[consecutiveLags] - 1,
  "<  (\:81ea\:7531\:5ea6 DOF = >", 2*(Length[consecutiveLags] - 1) + 1, "<)>"]
Print["<\:8fde\:7eed\:90e8\:5206: 0 \:5230 >", Max[consecutiveLags]]

(* DOF \:4e0e\:4f20\:611f\:5668\:6570\:5bf9\:6bd4 *)
Print["<ULA ", Length[TCA], "\:4f20\:611f\:5668\:7684 DOF = >", Length[TCA] - 1]
Print["<TCA  ", Length[TCA], "\:4f20\:611f\:5668\:7684 DOF = >",
  2*(Length[consecutiveLags] - 1) + 1, "<  (\:8fdc\:5927\:4e8e ULA!)>"]

(* ============================================================
   \:7b2c3\:8282: \:5bfc\:5411\:5411\:91cf\:4e0e\:5bfc\:5411\:77e9\:9635
   ============================================================ *)

(* \:5bfc\:5411\:5411\:91cf: a(\[Theta]) = exp(j*\[Pi]*p*sin(\[Theta]))
   \:6ce8\:610f: \:4f4d\:7f6e p \:4ee5 d=\[Lambda]/2 \:4e3a\:5355\:4f4d
   \:3010\:6027\:80fd\:5173\:952e\:3011N[] \:5f3a\:5236\:6570\:503c\:5316\:ff0c\:907f\:514d\:7b26\:53f7\:8fd0\:7b97 *)
a[theta_] := Exp[I * Pi * TCA * Sin[N[theta] * Degree]]

(* \:9a8c\:8bc1: theta=0 \:65f6\:6240\:6709\:5143\:7d20\:76f8\:4f4d\:4e3a0 *)
Print["<a(0\[Degree]) \:6a21\:503c = >", Abs[a[0]], "<  (\:5168\:4e3a1\:ff0c\:6b63\:786e)>"]
Print["<a(30\[Degree]) \:7b2c\:4e00\:5143\:7d20\:76f8\:4f4d = >", Arg[a[30][[1]]] * 180/Pi,
  "<\[Degree]  (\:5e94\:4e3a 0, \:56e0\:4e3a p0=0)>"]
Print["<a(30\[Degree]) \:7b2c\:4e8c\:5143\:7d20\:76f8\:4f4d = >", Arg[a[30][[2]]] * 180/Pi, "<\[Degree]>"]
Print["<  \:9a8c\:7b97: pi*5*sin(30\[Degree])*180/pi = >", 5*Sin[30 Degree]*180,
  "<\[Degree] mod 360 = >", Mod[5*Sin[30 Degree]*180, 360], "<\[Degree]>"]

(* DOA \:683c\:7f51 *)
doaGrid = N[Range[-60, 60, 1]];  (* -60\[Degree] \:5230 +60\[Degree]\:ff0c\:5171 121 \:4e2a\:89d2\:5ea6 *)
numClasses = Length[doaGrid];
Print["<DOA \:683c\:7f51: >", doaGrid[[{1, 2, 3}]], "< ... >",
  doaGrid[[{119, 120, 121}]], "<  \:5171>", numClasses, "<\:4e2a>"]

(* \:5bfc\:5411\:77e9\:9635 A \[Element] C^(P\[Times]121) *)
Apolarization = Table[a[doaGrid[[k]]], {k, numClasses}];
A = Transpose[Apolarization];  (* P \[Times] numClasses *)
Print["<\:5bfc\:5411\:77e9\:9635 A \:7ef4\:5ea6: >", Dimensions[A], "<  (P \[Times] numClasses)>"]

(* \:53ef\:89c6\:5316\:5bfc\:5411\:77e9\:9635\:7684\:76f8\:4f4d *)
MatrixPlot[Arg[A]/Pi,
  ColorFunction -> "Rainbow",
  FrameLabel -> {"\:4f20\:611f\:5668\:7d22\:5f15", "DOA \:89d2\:5ea6\:7d22\:5f15"},
  PlotLabel -> "\:5bfc\:5411\:77e9\:9635\:76f8\:4f4d (\[Times]\[Pi] rad)",
  ColorFunctionScaling -> False
]

(* ============================================================
   \:7b2c4\:8282: \:4fe1\:53f7\:6a21\:578b\:4eff\:771f
   ============================================================ *)

SeedRandom[42];
P     = Length[TCA];   (* \:4f20\:611f\:5668\:6570 = 12 *)
T     = 16;            (* \:5feb\:62cd\:6570 *)
K     = 3;             (* \:4fe1\:6e90\:6570 *)
snrDB = 10;            (* SNR (dB) *)
thetaTrue = {-20., 10., 35.};  (* \:771f\:5b9e DOA *)

(* \:4fe1\:6e90\:4fe1\:53f7: s(t) ~ CN(0, I)\:ff0c\:5f62\:72b6 K\[Times]T *)
S = RandomVariate[NormalDistribution[0, 1/Sqrt[2]], {K, T}] +
    I*RandomVariate[NormalDistribution[0, 1/Sqrt[2]], {K, T}];

(* \:8be5K\:4e2a\:4fe1\:6e90\:7684\:5bfc\:5411\:77e9\:9635 Ak \[Element] C^(P\[Times]K) *)
Ak = Transpose[Table[a[thetaTrue[[k]]], {k, K}]];
Print["<\:4fe1\:6e90\:5bfc\:5411\:77e9\:9635 Ak \:7ef4\:5ea6: >", Dimensions[Ak]]

(* \:5e72\:51c0\:4fe1\:53f7 Xclean = Ak . S\:ff0c\:5f62\:72b6 P\[Times]T *)
Xclean = Ak . S;

(* \:566a\:58f0\:529f\:7387\:8ba1\:7b97 *)
signalPower = Mean[Abs[Flatten[Xclean]]^2];
snrLinear   = 10^(snrDB/10.);
noiseStd    = Sqrt[signalPower / snrLinear / 2];
Print["<\:4fe1\:53f7\:529f\:7387 = >", signalPower]
Print["<\:566a\:58f0\:6807\:51c6\:5dee = >", noiseStd]

(* \:52a0\:6027\:9ad8\:65af\:767d\:566a\:58f0 *)
Noise = RandomVariate[NormalDistribution[0, noiseStd], {P, T}] +
        I*RandomVariate[NormalDistribution[0, noiseStd], {P, T}];

(* \:63a5\:6536\:4fe1\:53f7 X = Xclean + Noise\:ff0c\:5f62\:72b6 P\[Times]T *)
X = Xclean + Noise;
Print["<\:63a5\:6536\:4fe1\:53f7 X \:7ef4\:5ea6: >", Dimensions[X]]
Print["<\:5b9e\:9645 SNR = >",
  10*Log10[signalPower / Mean[Abs[Flatten[Noise]]^2]],
  "< dB  (\:671f\:671b >", snrDB, "< dB)>"]

(* ============================================================
   \:7b2c5\:8282: \:7406\:8bba\:534f\:65b9\:5dee\:77e9\:9635\:4e0e\:6837\:672c\:534f\:65b9\:5dee\:77e9\:9635
   ============================================================ *)

(* \:7406\:8bba: R = Ak*Rs*Ak^H + \[Sigma]_n\.b2*I *)
Rs      = IdentityMatrix[K];  (* \:5f52\:4e00\:5316\:4fe1\:6e90\:529f\:7387 *)
Rtheory = Ak . Rs . ConjugateTranspose[Ak] +
          noiseStd^2 * IdentityMatrix[P];

(* \:6837\:672c\:534f\:65b9\:5dee (MLE): R_hat = X*X^H / T *)
Rhat = X . ConjugateTranspose[X] / T;

Print["<\:7406\:8bba\:534f\:65b9\:5dee\:77e9\:9635\:7ef4\:5ea6: >", Dimensions[Rtheory],
  "<  Hermitian: >", Rtheory === ConjugateTranspose[Rtheory]]
Print["<\:6837\:672c\:534f\:65b9\:5dee\:77e9\:9635\:7ef4\:5ea6: >", Dimensions[Rhat]]

(* \:53ef\:89c6\:5316\:5b9e\:90e8\:548c\:865a\:90e8 *)
Grid[{{
  MatrixPlot[Re[Rhat], ColorFunction -> "Rainbow", PlotLabel -> "Re(R\:0302)"],
  MatrixPlot[Im[Rhat], ColorFunction -> "Rainbow", PlotLabel -> "Im(R\:0302)"]
}}]

(* ============================================================
   \:7b2c6\:8282: MUSIC \:7b97\:6cd5
   ============================================================ *)

(* \:7279\:5f81\:503c\:5206\:89e3 *)
{evals, evecs} = Eigensystem[Rhat];

(* \:6309\:7279\:5f81\:503c\:964d\:5e8f\:6392\:5217 *)
order = Reverse[Ordering[Re[evals]]];  (* \:5168\:90e8\:7279\:5f81\:503c\:964d\:5e8f\:7d22\:5f15 *)
evals = Re[evals[[order]]];
evecs = evecs[[order]];

Print["<\:7279\:5f81\:503c (\:964d\:5e8f): >", NumberForm[evals, 4]]
Print["<\:524dK\:4e2a\:7279\:5f81\:503c >> \:540e P-K \:4e2a\:7279\:5f81\:503c: \:533a\:5206\:4fe1\:53f7\:5b50\:7a7a\:95f4\:548c\:566a\:58f0\:5b50\:7a7a\:95f4>"]

(* \:566a\:58f0\:5b50\:7a7a\:95f4: \:6700\:5c0f\:7684 P-K \:4e2a\:7279\:5f81\:5411\:91cf (\:5217) *)
Un = Transpose[evecs[[K + 1 ;; P]]];  (* P \[Times] (P-K) *)
Print["<\:566a\:58f0\:5b50\:7a7a\:95f4 Un \:7ef4\:5ea6: >", Dimensions[Un]]

(* MUSIC \:4f2a\:8c31
   \:3010\:6027\:80fd\:5173\:952e1\:3011thetaRange \:7528 N[] \:5f3a\:5236\:6d6e\:70b9\:ff0c\:907f\:514d\:7b26\:53f7\:77e9\:9635\:8fd0\:7b97 *)
thetaRange    = N[Range[-90, 90, 0.1]];   (* 1801 \:4e2a\:70b9 *)
UnProj        = Un . ConjugateTranspose[Un];  (* \:9884\:8ba1\:7b97\:6295\:5f71\:77e9\:9635\:ff0c\:907f\:514d\:91cd\:590d *)

(* \:3010\:6027\:80fd\:5173\:952e2\:3011\:9884\:8ba1\:7b97 UnProj\:ff0cTable \:5185\:53ea\:505a\:5411\:91cf\:4e58\:6cd5 *)
musicSpectrum = Table[
  Block[{av = a[theta]},
    {theta, 10*Log10[1 / Re[ConjugateTranspose[av] . UnProj . av]]}
  ],
  {theta, thetaRange}
];

(* \:53ef\:89c6\:5316 MUSIC \:8c31 *)
ListLinePlot[musicSpectrum,
  PlotRange -> All,
  AxesLabel -> {"\[Theta] (\:5ea6)", "P_MUSIC (dB)"},
  PlotLabel -> "MUSIC \:4f2a\:8c31  (\:771f\:5b9e DOA: " <> ToString[thetaTrue] <> "\[Degree])",
  Epilog -> {Red, Dashed,
    (Line[{{#, -100}, {#, 200}}] &) /@ thetaTrue},
  GridLines -> {thetaTrue, Automatic},
  ImageSize -> 600
]

(* \:5cf0\:503c\:68c0\:6d4b *)
spectVals = musicSpectrum[[All, 2]];
peaks = {};
Do[
  If[i > 1 && i < Length[spectVals] &&
     spectVals[[i]] > spectVals[[i - 1]] &&
     spectVals[[i]] > spectVals[[i + 1]] &&
     spectVals[[i]] > Mean[spectVals] + 3*StandardDeviation[spectVals],
    AppendTo[peaks, musicSpectrum[[i, 1]]]
  ],
  {i, Length[spectVals]}
];
peaks = Sort[peaks];
Print["<MUSIC \:4f30\:8ba1 DOA = >", peaks]
Print["<\:771f\:5b9e DOA       = >", thetaTrue]
Print["<\:8bef\:5dee (\[Degree])       = >", peaks - thetaTrue]

(* ============================================================
   \:7b2c7\:8282: BCELoss \:6570\:5b66\:63a8\:5bfc
   ============================================================ *)

(* Sigmoid \:51fd\:6570 *)
sigmoid[z_] := 1 / (1 + Exp[-z])

(* \:53ef\:89c6\:5316 sigmoid *)
Plot[sigmoid[z], {z, -6, 6},
  AxesLabel -> {"z", "\[Sigma](z)"},
  PlotLabel -> "Sigmoid \:6fc0\:6d3b\:51fd\:6570",
  PlotStyle -> Blue,
  GridLines -> {{0}, {0, 0.5, 1}},
  ImageSize -> 400
]

(* BCELoss \:5bf9\:5355\:4e2a\:6837\:672c: y \[Element] {0,1}, \:0177 \[Element] (0,1) *)
BCELoss[y_, yhat_] := -(y*Log[yhat] + (1 - y)*Log[1 - yhat])

(* BCELoss \:53ef\:89c6\:5316 *)
Plot[{BCELoss[1, p], BCELoss[0, p]}, {p, 0.001, 0.999},
  PlotLegends -> {"y=1 (\:6709\:4fe1\:6e90)", "y=0 (\:65e0\:4fe1\:6e90)"},
  AxesLabel -> {"\:9884\:6d4b\:6982\:7387 \:0177", "BCE Loss"},
  PlotLabel -> "Binary Cross-Entropy Loss",
  PlotStyle -> {Blue, Red},
  ImageSize -> 500
]

(* \:68af\:5ea6: d(BCE)/dz = \[Sigma](z) - y = \:0177 - y *)
Print["<BCE \:5bf9 z \:7684\:68af\:5ea6 = \[Sigma](z) - y = \:0177 - y>"]
dBCEdz[y_, z_] := sigmoid[z] - y
Print["<\:9a8c\:8bc1: y=1, z=2: \:68af\:5ea6 = >", dBCEdz[1, 2.0],
  "<  (\:5e94\:63a5\:8fd10\:ff0c\:56e0\:4e3a sigmoid(2)\[TildeTilde]0.88 \:63a5\:8fd11)>"]
Print["<\:9a8c\:8bc1: y=0, z=2: \:68af\:5ea6 = >", dBCEdz[0, 2.0],
  "<  (\:5e94\:4e3a\:6b63\:ff0c\:9700\:8981\:964d\:4f4e z)>"]

(* ============================================================
   \:7b2c8\:8282: Cram\[EAcute]r-Rao Bound (CRB) \:7406\:8bba\:4e0b\:754c
   ============================================================ *)

(* CRB \:7ed9\:51fa DOA \:4f30\:8ba1\:65b9\:5dee\:7684\:7406\:8bba\:4e0b\:754c
   \:5bf9\:4e8e ULA\:ff0c\:7b2c k \:4e2a\:4fe1\:6e90\:7684 CRB:
   CRB(\[Theta]_k) = 6 / (SNR * T * \[Pi]\.b2 * cos\.b2(\[Theta]_k) * (P\.b3 - P)) *)
crbULA[thetaRad_, snrLinear_, nT_, nP_] :=
    6 / (snrLinear * nT * Pi^2 * Cos[thetaRad]^2 * (nP^3 - nP))

(* \:4ee5\:5ea6\:4e3a\:5355\:4f4d\:7684 RMSE \:4e0b\:754c *)
crbDeg[thetaDeg_, snrDB_, nT_, nP_] :=
    Sqrt[crbULA[thetaDeg*Degree, 10^(snrDB/10), nT, nP]] * 180/Pi

(* SNR vs RMSE \:66f2\:7ebf *)
snrRange = N[Range[-5, 25, 1]];
crbCurve = Table[{snr, crbDeg[0, snr, 16, 12]}, {snr, snrRange}];

ListLinePlot[crbCurve,
  AxesLabel -> {"SNR (dB)", "RMSE \:4e0b\:754c (\[Degree])"},
  PlotLabel -> "CRB: \[Theta]=0\[Degree], T=16, P=12 (ULA \:8fd1\:4f3c)",
  ScalingFunctions -> {"Linear", "Log"},
  GridLines -> {{0, 5, 10, 15, 20}, Automatic},
  ImageSize -> 500
]

(* \:4e0d\:540c T \:503c\:7684\:5bf9\:6bd4 *)
Plot[{crbDeg[0, snr, 16, 12], crbDeg[0, snr, 32, 12]},
  {snr, -5, 25},
  PlotLegends -> {"T=16", "T=32"},
  AxesLabel -> {"SNR (dB)", "RMSE \:4e0b\:754c (\[Degree])"},
  PlotLabel -> "CRB vs SNR (\:4e0d\:540c\:5feb\:62cd\:6570 T)",
  ScalingFunctions -> {"Linear", "Log"},
  ImageSize -> 500
]

(* ============================================================
   \:7b2c9\:8282: \:8bc4\:4f30\:6307\:6807\:8ba1\:7b97\:793a\:4f8b
   ============================================================ *)

SeedRandom[100];
numSamples  = 100;
numClasses2 = 121;
threshold   = 0.5;

(* \:968f\:673a\:751f\:6210\:771f\:5b9e\:6807\:7b7e (\:7a00\:758f\:ff0c\:6bcf\:4e2a\:6837\:672c\:7ea6 K=3 \:4e2a\:6fc0\:6d3b\:7c7b\:522b) *)
yTrue = Table[
  Module[{label = ConstantArray[0, numClasses2]},
    Scan[(label[[#]] = 1) &, RandomSample[Range[numClasses2], 3]];
    label
  ],
  {numSamples}
];

(* \:6a21\:62df\:9884\:6d4b\:6982\:7387 (\:63a5\:8fd1\:771f\:5b9e\:6807\:7b7e\:ff0c\:52a0\:4e00\:4e9b\:566a\:58f0) *)
yPred    = Clip[yTrue + RandomVariate[NormalDistribution[0, 0.3],
              {numSamples, numClasses2}], {0, 1}];
yPredBin = Round[yPred - threshold + 0.5];  (* \:9608\:503c\:5316 *)

(* \:8ba1\:7b97 TP, FP, TN, FN *)
TP = Total[Flatten[yTrue * yPredBin]];
FP = Total[Flatten[(1 - yTrue) * yPredBin]];
TN = Total[Flatten[(1 - yTrue) * (1 - yPredBin)]];
FN = Total[Flatten[yTrue * (1 - yPredBin)]];

Print["<TP = >", TP, "<  FP = >", FP, "<  TN = >", TN, "<  FN = >", FN]
Print["<Accuracy    = >", N[(TP + TN)/(TP + FP + TN + FN)]]
Print["<Precision   = >", N[TP/(TP + FP)]]
Print["<Recall      = >", N[TP/(TP + FN)]]
Print["<Specificity = >", N[TN/(TN + FP)]]
Print["<F1 Score    = >", N[2*TP/(2*TP + FP + FN)]]

(* ============================================================
   \:7b2c10\:8282: \:9635\:5217\:6270\:52a8\:5b9e\:9a8c\:6570\:5b66\:57fa\:7840 (Extension)
   ============================================================ *)

epsilon      = 0.05;  (* 5% \:6807\:51c6\:5dee *)
perturbedTCA = TCA + RandomVariate[NormalDistribution[0, epsilon], Length[TCA]];
Print["<\:539f\:59cb TCA = >", TCA]
Print["<\:6270\:52a8 TCA = >", N[perturbedTCA, 3]]

(* \:6270\:52a8\:540e\:7684\:5bfc\:5411\:5411\:91cf *)
aPert[theta_, positions_] := Exp[I * Pi * positions * N[Sin[theta * Degree]]]

(* \:5931\:914d\:8bef\:5dee: |a(\[Theta]) - a_pert(\[Theta])|\.b2 / P *)
mismatch[theta_, eps_] := Module[{dpos = RandomVariate[NormalDistribution[0, eps], Length[TCA]]},
  Norm[a[theta] - aPert[theta, TCA + dpos]]^2 / Length[TCA]
]

(* \:5bf9\:591a\:4e2a epsilon \:503c\:8ba1\:7b97\:671f\:671b\:5931\:914d *)
epsilonRange = {0.01, 0.05, 0.10, 0.20};
thetaTest    = 0.;
Table[
  Block[{avgMismatch = Mean[Table[mismatch[thetaTest, eps], {100}]]},
    Print["<epsilon = >", eps, "<  \:5e73\:5747\:5931\:914d = >", N[avgMismatch, 4]]
  ],
  {eps, epsilonRange}
];

(* ============================================================
   \:7b2c11\:8282: \:5dee\:5206\:9635\:5217\:534f\:65b9\:5dee\:5411\:91cf\:5316 (\:7528\:4e8e\:6b20\:5b9a DOA \:4f30\:8ba1)
   ============================================================ *)

(* \:5411\:91cf\:5316\:6837\:672c\:534f\:65b9\:5dee\:77e9\:9635 *)
rVec = Flatten[Transpose[Rhat]];  (* vec(R_hat): P\.b2 \[Times] 1 *)

(* \:5dee\:5206\:9635\:5217\:5bf9\:5e94\:7684\:884c\:9009\:62e9\:77e9\:9635 J
   \:6bcf\:4e2a\:5dee\:5206 p_i - p_j \:5bf9\:5e94 R_hat[i,j] = rVec[(j-1)*P + i] *)
diffPairs    = Flatten[Table[{i, j}, {i, P}, {j, P}], 1];
uniqueDiffs  = Union[Map[(TCA[[#[[1]]]] - TCA[[#[[2]]]]) &, diffPairs]];

(* \:3010\:6027\:80fd\:4f18\:5316\:3011\:7528 GroupBy \:4ee3\:66ff Select \:5faa\:73af\:ff0c\:901f\:5ea6\:63d0\:5347 ~10\[Times] *)
buildVirtualCovariance[rHat_] := Module[
  {grouped = GroupBy[diffPairs, (TCA[[#[[1]]]] - TCA[[#[[2]]]]) &]},
  Table[
    Mean[Map[(rHat[[#[[1]], #[[2]]]] &), grouped[uniqueDiffs[[l]]]]],
    {l, Length[uniqueDiffs]}
  ]
];

rVirtual = buildVirtualCovariance[Rhat];
Print["<\:865a\:62df\:5dee\:5206\:9635\:5217\:957f\:5ea6 = >", Length[uniqueDiffs],
  "<  (\:63d0\:4f9b >", Length[uniqueDiffs], "< \:4e2a\:865a\:62df\:4f20\:611f\:5668)>"]

(* ============================================================
   \:7b2c12\:8282: SNR \:6027\:80fd\:66f2\:7ebf (\:9884\:671f\:5f62\:72b6)
   ============================================================ *)

(* \:6a21\:62df CNN \:6027\:80fd\:66f2\:7ebf (\:793a\:610f\:ff0c\:57fa\:4e8e\:8bba\:6587 Fig.4-5 \:7684\:5178\:578b\:503c) *)
snrValues  = {-5, 0, 5, 10, 15, 20};
recallRaw16  = {0.55, 0.72, 0.85, 0.92, 0.96, 0.98};
recallRaw32  = {0.62, 0.78, 0.88, 0.94, 0.97, 0.99};
recallCov16  = {0.60, 0.75, 0.87, 0.93, 0.97, 0.98};
recallCov32  = {0.65, 0.80, 0.90, 0.95, 0.98, 0.99};
recallMUSIC  = {0.30, 0.52, 0.70, 0.85, 0.92, 0.96};
recallESPRIT = {0.20, 0.40, 0.62, 0.80, 0.89, 0.94};

ListLinePlot[
  {
    Transpose[{snrValues, recallRaw16}],
    Transpose[{snrValues, recallRaw32}],
    Transpose[{snrValues, recallCov16}],
    Transpose[{snrValues, recallCov32}],
    Transpose[{snrValues, recallMUSIC}],
    Transpose[{snrValues, recallESPRIT}]
  },
  PlotLegends -> {"CNN Raw T=16", "CNN Raw T=32",
                  "CNN Cov T=16", "CNN Cov T=32",
                  "MUSIC", "ESPRIT"},
  AxesLabel -> {"SNR (dB)", "Recall"},
  PlotLabel -> "\:6027\:80fd vs SNR (\:793a\:610f\:66f2\:7ebf - \:5b9e\:9645\:503c\:5f85\:8bad\:7ec3\:5b8c\:6210\:540e\:66f4\:65b0)",
  PlotStyle -> {Blue, Cyan, Orange, Red, Green, Purple},
  PlotMarkers -> {"\[FilledCircle]", "\[FilledSquare]",
                  "\[FilledDiamond]", "\[FilledUpTriangle]",
                  "\[FilledCircle]", "\[FilledSquare]"},
  PlotRange -> {0, 1},
  GridLines -> Automatic,
  ImageSize -> 600
]

(* ============================================================
   \:8f93\:51fa\:603b\:7ed3
   ============================================================ *)
Print[""]
Print["<=== \:6f14\:7b97\:5b8c\:6210 ===>"]
Print["<TCA: M=", M, ", N=", N, ", P=", P, "\:4f20\:611f\:5668, \:5b54\:5f84=", Max[TCA], "d>"]
Print["<DOF (\:865a\:62df): >", 2*(Length[consecutiveLags] - 1) + 1,
  "< vs ULA DOF: >", P - 1]
Print["<MUSIC \:4f30\:8ba1\:8bef\:5dee (\[Degree]): >", N[peaks - thetaTrue, 3]]



