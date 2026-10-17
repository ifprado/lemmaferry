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
