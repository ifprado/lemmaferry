-- Generated constructive propositional examples. Run Lean to verify.
set_option autoImplicit false

theorem lf_validation_000000
    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 : Prop)
    (h_start : p6)
    (h_e0 : p5 → p2)
    (h_e1 : p6 → p9)
    (h_e2 : p2 → p1)
    (h_e3 : p9 → p13)
    (h_e4 : p4 → p16)
    (h_e5 : p13 → p4)
    (h_e6 : p8 → p15)
    (h_e7 : p16 → p1)
    (h_e8 : p0 → p10)
    (h_e9 : p7 → p1)
    (h_e10 : p3 → p12)
    (h_e11 : p6 → p5)
    (h_e12 : p11 → p14)
    (h_e13 : p12 → p7)
    (h_e14 : p6 → p3)
    : p1 :=
  h_e2 (h_e0 (h_e11 (h_start)))

#print axioms lf_validation_000000
theorem lf_validation_000000_long
    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 : Prop)
    (h_start : p6)
    (h_e0 : p5 → p2)
    (h_e1 : p6 → p9)
    (h_e2 : p2 → p1)
    (h_e3 : p9 → p13)
    (h_e4 : p4 → p16)
    (h_e5 : p13 → p4)
    (h_e6 : p8 → p15)
    (h_e7 : p16 → p1)
    (h_e8 : p0 → p10)
    (h_e9 : p7 → p1)
    (h_e10 : p3 → p12)
    (h_e11 : p6 → p5)
    (h_e12 : p11 → p14)
    (h_e13 : p12 → p7)
    (h_e14 : p6 → p3)
    : p1 :=
  h_e7 (h_e4 (h_e5 (h_e3 (h_e1 (h_start)))))

#print axioms lf_validation_000000_long

theorem lf_validation_000001
    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 : Prop)
    (h_start : p4)
    (h_e0 : p7 → p5)
    (h_e1 : p1 → p9)
    (h_e2 : p8 → p10)
    (h_e3 : p9 → p7)
    (h_e4 : p2 → p8)
    (h_e5 : p6 → p11)
    (h_e6 : p4 → p2)
    (h_e7 : p3 → p0)
    (h_e8 : p4 → p1)
    (h_e9 : p4 → p3)
    (h_e10 : p0 → p12)
    (h_e11 : p12 → p10)
    (h_e12 : p5 → p10)
    : p10 :=
  h_e2 (h_e4 (h_e6 (h_start)))

#print axioms lf_validation_000001
theorem lf_validation_000001_long
    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 : Prop)
    (h_start : p4)
    (h_e0 : p7 → p5)
    (h_e1 : p1 → p9)
    (h_e2 : p8 → p10)
    (h_e3 : p9 → p7)
    (h_e4 : p2 → p8)
    (h_e5 : p6 → p11)
    (h_e6 : p4 → p2)
    (h_e7 : p3 → p0)
    (h_e8 : p4 → p1)
    (h_e9 : p4 → p3)
    (h_e10 : p0 → p12)
    (h_e11 : p12 → p10)
    (h_e12 : p5 → p10)
    : p10 :=
  h_e12 (h_e0 (h_e3 (h_e1 (h_e8 (h_start)))))

#print axioms lf_validation_000001_long

theorem lf_validation_000002
    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 : Prop)
    (h_start : p5)
    (h_e0 : p12 → p3)
    (h_e1 : p5 → p12)
    (h_e2 : p2 → p8)
    (h_e3 : p7 → p8)
    (h_e4 : p9 → p8)
    (h_e5 : p1 → p4)
    (h_e6 : p5 → p11)
    (h_e7 : p6 → p0)
    (h_e8 : p10 → p7)
    (h_e9 : p4 → p9)
    (h_e10 : p3 → p2)
    (h_e11 : p11 → p1)
    (h_e12 : p5 → p10)
    : p8 :=
  h_e3 (h_e8 (h_e12 (h_start)))

#print axioms lf_validation_000002
theorem lf_validation_000002_long
    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 : Prop)
    (h_start : p5)
    (h_e0 : p12 → p3)
    (h_e1 : p5 → p12)
    (h_e2 : p2 → p8)
    (h_e3 : p7 → p8)
    (h_e4 : p9 → p8)
    (h_e5 : p1 → p4)
    (h_e6 : p5 → p11)
    (h_e7 : p6 → p0)
    (h_e8 : p10 → p7)
    (h_e9 : p4 → p9)
    (h_e10 : p3 → p2)
    (h_e11 : p11 → p1)
    (h_e12 : p5 → p10)
    : p8 :=
  h_e4 (h_e9 (h_e5 (h_e11 (h_e6 (h_start)))))

#print axioms lf_validation_000002_long
