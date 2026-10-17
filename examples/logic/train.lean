-- Generated constructive propositional examples. Run Lean to verify.
set_option autoImplicit false

theorem lf_train_000000
    (p0 p1 p2 p3 p4 p5 p6 p7 : Prop)
    (h_start : p4)
    (h_e0 : p4 → p6)
    (h_e1 : p2 → p1)
    (h_e2 : p4 → p5)
    (h_e3 : p3 → p0)
    (h_e4 : p6 → p7)
    (h_e5 : p5 → p0)
    (h_e6 : p7 → p3)
    : p0 :=
  h_e5 (h_e2 (h_start))

#print axioms lf_train_000000
theorem lf_train_000000_long
    (p0 p1 p2 p3 p4 p5 p6 p7 : Prop)
    (h_start : p4)
    (h_e0 : p4 → p6)
    (h_e1 : p2 → p1)
    (h_e2 : p4 → p5)
    (h_e3 : p3 → p0)
    (h_e4 : p6 → p7)
    (h_e5 : p5 → p0)
    (h_e6 : p7 → p3)
    : p0 :=
  h_e3 (h_e6 (h_e4 (h_e0 (h_start))))

#print axioms lf_train_000000_long

theorem lf_train_000001
    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 : Prop)
    (h_start : p3)
    (h_e0 : p9 → p5)
    (h_e1 : p1 → p5)
    (h_e2 : p7 → p0)
    (h_e3 : p0 → p1)
    (h_e4 : p6 → p4)
    (h_e5 : p3 → p7)
    (h_e6 : p3 → p9)
    (h_e7 : p2 → p8)
    : p5 :=
  h_e0 (h_e6 (h_start))

#print axioms lf_train_000001
theorem lf_train_000001_long
    (p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 : Prop)
    (h_start : p3)
    (h_e0 : p9 → p5)
    (h_e1 : p1 → p5)
    (h_e2 : p7 → p0)
    (h_e3 : p0 → p1)
    (h_e4 : p6 → p4)
    (h_e5 : p3 → p7)
    (h_e6 : p3 → p9)
    (h_e7 : p2 → p8)
    : p5 :=
  h_e1 (h_e3 (h_e2 (h_e5 (h_start))))

#print axioms lf_train_000001_long

theorem lf_train_000002
    (p0 p1 p2 p3 p4 p5 p6 p7 : Prop)
    (h_start : p4)
    (h_e0 : p3 → p1)
    (h_e1 : p0 → p6)
    (h_e2 : p4 → p5)
    (h_e3 : p7 → p2)
    (h_e4 : p4 → p3)
    (h_e5 : p1 → p7)
    (h_e6 : p5 → p2)
    : p2 :=
  h_e6 (h_e2 (h_start))

#print axioms lf_train_000002
theorem lf_train_000002_long
    (p0 p1 p2 p3 p4 p5 p6 p7 : Prop)
    (h_start : p4)
    (h_e0 : p3 → p1)
    (h_e1 : p0 → p6)
    (h_e2 : p4 → p5)
    (h_e3 : p7 → p2)
    (h_e4 : p4 → p3)
    (h_e5 : p1 → p7)
    (h_e6 : p5 → p2)
    : p2 :=
  h_e3 (h_e5 (h_e0 (h_e4 (h_start))))

#print axioms lf_train_000002_long
