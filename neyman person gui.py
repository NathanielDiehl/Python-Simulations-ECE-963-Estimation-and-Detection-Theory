
# auther: Nathan Diehl
# class: ECE 963 - Estimation and Detection Theory


from detection_theory import *




## initialize parameters
mu0_0   = -0.25
mu1_0   =  0.25
sigma0_0 =  0.2
sigma1_0 =  0.2
pi_0       = 0.5
alpha_0    = 0.5
cost       = [[0,1],[1,0]]
tau = 0
tau_prime  = 0
t = np.arange(-1.0, 1.0, 0.001)

def update_tau(pi_0, cost, pdf0_, pdf1_):
    global pdf0
    global pdf1
    global tau;       tau        = pi_0/(1-pi_0)*(cost[1][0] - cost[0][0])/(cost[0][1] - cost[1][1])
    global tau_prime; tau_prime  = get_np_tau_prime(cost, mu0_0, mu1_0, sigma0_0, pdf0, pdf1, alpha_0)


## define pdf functions
pdf0 = lambda y_: normal(mu0_0, sigma0_0, y_)
pdf1 = lambda y_: normal(mu1_0, sigma1_0, y_)

update_tau(pi_0, cost, pdf0, pdf1)

## set up figure
axis_color = 'lightgoldenrodyellow'
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.40, right=0.75)
xlim_min = -1
xlim_max = 1
ylim_min = -1
ylim_max = 5

[pdf0_line]     = ax.plot(t, pdf0(t), linewidth=2, color='red', alpha=0.5)
pdf0_fill       = ax.fill_between(t, pdf0(t), color='pink', alpha=0.2)
[pdf1_line]     = ax.plot(t, pdf1(t), linewidth=2, color='darkblue', alpha=0.5)
pdf1_fill       = ax.fill_between(t, pdf1(t), color='lightblue', alpha=0.2)
pdf0_name  = ax.text(mu0_0 - 0.1, 0.25+pdf0(mu0_0), r'$p_0(y)$', fontsize=11)
pdf1_name  = ax.text(mu1_0 - 0.1, 0.25+pdf1(mu1_0), r'$p_1(y)$', fontsize=11)
[decision] = ax.plot([tau_prime]*2, [ylim_min, ylim_max], linewidth=2, color='lightgreen', linestyle='--')
ax.set_xlim([xlim_min, xlim_max])
ax.set_ylim([ylim_min, ylim_max])

def draw_plots():   
    pdf0_line.set_ydata(pdf0(t))
    global pdf0_fill; pdf0_fill.remove()
    pdf0_fill       = ax.fill_between(t, pdf0(t), color='pink', alpha=0.2)
    # print(np.column_stack((t, pdf0(t))))
    # pdf0_fill.set_paths(np.column_stack((t, pdf0(t))) ) #.set_aa(np.column_stack((t, pdf0(t))))
    pdf1_line.set_ydata(pdf1(t))
    global pdf1_fill; pdf1_fill.remove()
    pdf1_fill       = ax.fill_between(t, pdf1(t), color='lightblue', alpha=0.2)
    # pdf1_fill.set_aa(np.column_stack((t, pdf1(t))))
    pdf0_name.set_position((mu0_0 - 0.1, 0.25+pdf0(mu0_0)))
    pdf1_name.set_position((mu1_0 - 0.1, 0.25+pdf1(mu1_0)))
    
    global tau_prime
    P_y = get_probability(cost, mu0_0, mu1_0, sigma0_0, tau_prime, pdf0, pdf1)
    decision.set_xdata([tau_prime]*2)

 
    P00_text.set_text(r'           $   P_{0}(\Gamma_0)$  = '   + f"{P_y[0][0]:0.3f}")
    P01_text.set_text(r'$P_F(\delta) = P_{0}(\Gamma_1)$  = '   + f"{P_y[0][1]:0.3f}")
    P10_text.set_text(r'$P_M(\delta) = P_{1}(\Gamma_0)$  = '   + f"{P_y[1][0]:0.3f}")
    P11_text.set_text(r'$P_D(\delta) = P_{1}(\Gamma_1)$  = '   + f"{P_y[1][1]:0.3f}")

    R_d = get_conditional_risk(cost, mu0_0, mu1_0, sigma0_0, tau_prime, pdf0, pdf1)
    r_d = get_baysian_risk(cost, mu0_0, mu1_0, sigma1_0, tau_prime, pi_0, pdf0, pdf1)
    R0_text.set_text(r'$R_0(\delta)$  = '   + f"{R_d[0]:0.3f}")
    R1_text.set_text(r'$R_1(\delta)$  = '   + f"{R_d[1]:0.3f}")
    rb_text.set_text(r'$r_b(\delta)$  = '   + f"{r_d:0.3f}")

    fig.canvas.draw_idle()




# Sliders
mu0_slider_ax    = fig.add_axes([0.25, 0.30, 0.45, 0.03], facecolor=axis_color)
mu0_slider       = Slider(mu0_slider_ax, r'$\mu_0$', -1.0, 1.0, valinit=mu0_0)

mu1_slider_ax    = fig.add_axes([0.25, 0.25, 0.45, 0.03], facecolor=axis_color)
mu1_slider       = Slider(mu1_slider_ax, r'$\mu_1$', -1.0, 1.0, valinit=mu1_0)

sigma0_slider_ax = fig.add_axes([0.25, 0.20, 0.45, 0.03], facecolor=axis_color)
sigma0_slider    = Slider(sigma0_slider_ax, r'$\sigma_0$', 0.01, 1.0, valinit=sigma0_0)

sigma1_slider_ax = fig.add_axes([0.25, 0.15, 0.45, 0.03], facecolor=axis_color)
sigma1_slider    = Slider(sigma1_slider_ax, r'$\sigma_1$', 0.01, 1.0, valinit=sigma1_0)

alpha_slider_ax     = fig.add_axes([0.25, 0.1, 0.45, 0.03], facecolor=axis_color)
alpha_slider        = Slider(alpha_slider_ax, r'$\alpha$', 0.01, 0.99, valinit=alpha_0)


P00_text = fig.text(0.775, 0.85, r'              $P_{0}(\Gamma_0)$  = ',   color='black', fontsize=10, visible=True)
P01_text = fig.text(0.775, 0.80, r'$P_F(\delta) = P_{0}(\Gamma_1)$  = ',   color='black', fontsize=10, visible=True)
P10_text = fig.text(0.775, 0.75, r'$P_M(\delta) = P_{1}(\Gamma_0)$  = ', color='black', fontsize=10, visible=True)
P11_text = fig.text(0.775, 0.70, r'$P_D(\delta) = P_{1}(\Gamma_1)$  = ', color='black', fontsize=10, visible=True)

R0_text  = fig.text(0.8, 0.60, r'$R_0(\delta)$  = ',   color='black', fontsize=10, visible=True)
R1_text  = fig.text(0.8, 0.55, r'$R_1(\delta)$  = ',   color='black', fontsize=10, visible=True)
rb_text  = fig.text(0.8, 0.50, r'$r_b(\delta)$  = ', color='black', fontsize=10, visible=True)


def sliders_on_changed(val):
    global mu0_0;   mu0_0     = mu0_slider.val
    global mu1_0;   mu1_0     = mu1_slider.val
    global sigma0_0; sigma0_0 = sigma0_slider.val
    global sigma1_0; sigma1_0 = sigma1_slider.val
    global alpha_0;    alpha_0      = alpha_slider.val

    update_tau(pi_0, cost, pdf0, pdf1)
    draw_plots()
mu0_slider.on_changed(sliders_on_changed)
mu1_slider.on_changed(sliders_on_changed)
sigma0_slider.on_changed(sliders_on_changed)
sigma1_slider.on_changed(sliders_on_changed)
alpha_slider.on_changed(sliders_on_changed)
sliders_on_changed(None)




reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    sigma0_slider.reset()
    mu0_slider.reset()
    mu1_slider.reset()
    alpha_slider.reset()
    CBox00.set_val(0)
    CBox01.set_val(1)
    CBox10.set_val(1)
    CBox11.set_val(0)
reset_button.on_clicked(reset_button_on_clicked)


pdf0_radios_ax_text = fig.text(     0.025, 0.7, r'$P_{0}(y)$ = ',   color='black', fontsize=10, visible=True)
pdf0_radios_ax      = fig.add_axes([0.025, 0.55, 0.135, 0.125], facecolor=axis_color)
pdf0_radios         = RadioButtons(pdf0_radios_ax, ('normal', 'laplacian', 'uniform'), active=0)

pdf0_radios_ax_text = fig.text(     0.025, 0.5, r'$P_{1}(y)$ = ',   color='black', fontsize=10, visible=True)
pdf1_radios_ax      = fig.add_axes([0.025, 0.35, 0.135, 0.125], facecolor=axis_color)
pdf1_radios         = RadioButtons(pdf1_radios_ax, ('normal', 'laplacian', 'uniform'), active=0)

def pdf0_radios_on_clicked(label):
    global pdf0
    if label == 'normal':
        pdf0 = lambda y_: normal(mu0_0, sigma0_0, y_)
    elif label == 'laplacian':
        pdf0 = lambda y_: laplacian(mu0_0, sigma0_0, y_)
    elif label == 'uniform':
        pdf0 = lambda y_: uniform(mu0_0, sigma0_0, y_)
    update_tau(pi_0, cost, pdf0, pdf1)
    draw_plots()
def pdf1_radios_on_clicked(label):
    global pdf1; 
    if label == 'normal':
        pdf1 = lambda y_: normal(mu1_0, sigma1_0, y_)
    elif label == 'laplacian':
        pdf1 = lambda y_: laplacian(mu1_0, sigma1_0, y_)
    elif label == 'uniform':
        pdf1 = lambda y_: uniform(mu1_0, sigma1_0, y_)
    update_tau(pi_0, cost, pdf0, pdf1)
    draw_plots()
pdf0_radios.on_clicked(pdf0_radios_on_clicked)
pdf1_radios.on_clicked(pdf1_radios_on_clicked)

def txtBox_on_changed(expression, i, j):
    cost[i][j] = float(expression)
    update_tau(pi_0, cost, pdf0, pdf1)
    draw_plots()
C_text = fig.text(0.025, 0.85, r'$C_{ij}$ = ',   color='black', fontsize=10, visible=True)
CBox01 = TextBox(fig.add_axes([0.15, 0.85, 0.045, 0.04]), "")
CBox00 = TextBox(fig.add_axes([0.1,  0.85, 0.045, 0.04]), "")
CBox11 = TextBox(fig.add_axes([0.15, 0.80, 0.045, 0.04]), "")
CBox10 = TextBox(fig.add_axes([0.1,  0.80, 0.045, 0.04]), "")
CBox00.on_submit(lambda x : txtBox_on_changed(x, 0, 0))
CBox01.on_submit(lambda x : txtBox_on_changed(x, 0, 1))
CBox10.on_submit(lambda x : txtBox_on_changed(x, 1, 0))
CBox11.on_submit(lambda x : txtBox_on_changed(x, 1, 1))

CBox00.set_val("0")
CBox11.set_val("0")
CBox01.set_val("1")
CBox10.set_val("1")

plt.show()