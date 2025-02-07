
# auther: Nathan Diehl
# class: ECE 963 - Estimation and Detection Theory


from detection_theory import *



## initialize parameters
mu0_0      = -0.15
mu1_0      =  0.25
sigma0_0   =  0.5
sigma1_0   =  0.3
tau_prime  = 0.0
pi_0       = 0.5
cost       = [[0,1],[1,0]]
t          = np.arange(-1.0, 1.0, 0.001)
tau_t      = np.arange(-1.0, 1.0, 0.01) 

## define pdf functions
pdf0 = lambda y_: normal(mu0_0, sigma0_0, y_)
pdf1 = lambda y_: normal(mu1_0, sigma1_0, y_)





## set up figure
axis_color = 'lightgoldenrodyellow'
fig, ax = plt.subplots(1, 2)
ax[0].set_title('pdf')
ax[1].set_title('risk')

fig.subplots_adjust(left=0.25, bottom=0.40)
xlim_min = -1
xlim_max = 1
ylim_min = -1
ylim_max = 10

[risk_line]             = ax[1].plot(np.arange(0, 1, 0.01), get_baysian_risk_plot(cost, mu0_0, mu1_0, sigma1_0, tau_prime, pdf0, pdf1), linewidth=2, color='red', alpha=0.5, label=r'$r(\pi_0,\delta)$')
[baysian_risk_line]     = ax[1].plot(np.arange(0, 1, 0.05), get_min_baysian_risk_plot(cost, mu0_0, mu1_0, sigma1_0, pdf0, pdf1), label=r'$V(\pi_0)$')
ax[1].legend(loc='upper right')



[pdf0_line]           = ax[0].plot(t, pdf0(t), linewidth=2, color='red', alpha=0.5)
pdf0_fill             = ax[0].fill_between(t, pdf0(t), color='pink', alpha=0.2)
[pdf1_line]           = ax[0].plot(t, pdf1(t), linewidth=2, color='darkblue', alpha=0.5)
pdf1_fill             = ax[0].fill_between(t, pdf1(t), color='lightblue', alpha=0.2)
pdf0_name  = ax[0].text(mu0_0 - 0.1, 0.25+pdf0(mu0_0), r'$p_0(y)$', fontsize=11)
pdf1_name  = ax[0].text(mu1_0 - 0.1, 0.25+pdf1(mu1_0), r'$p_1(y)$', fontsize=11)
[decision] = ax[0].plot([tau_prime]*2, [ylim_min, ylim_max], linewidth=2, color='lightgreen', linestyle='--')
ax[0].set_xlim([xlim_min, xlim_max])
ax[0].set_ylim([ylim_min, ylim_max])
ax[1].set_ylim([0, 1])

def draw_plots():   
    pdf0_line.set_ydata(pdf0(t))
    global pdf0_fill; pdf0_fill.remove()
    pdf0_fill       = ax[0].fill_between(t, pdf0(t), color='pink', alpha=0.2)
    
    pdf1_line.set_ydata(pdf1(t))
    global pdf1_fill; pdf1_fill.remove()
    pdf1_fill       = ax[0].fill_between(t, pdf1(t), color='lightblue', alpha=0.2)

    global tau_prime
    risk_line.set_ydata(get_baysian_risk_plot(cost, mu0_0, mu1_0, sigma1_0, tau_prime, pdf0, pdf1))
    baysian_risk_line.set_ydata(get_min_baysian_risk_plot(cost, mu0_0, mu1_0, sigma1_0, pdf0, pdf1))

    pdf0_name.set_position((mu0_0 - 0.1, 0.25+pdf0(mu0_0)))
    pdf1_name.set_position((mu1_0 - 0.1, 0.25+pdf1(mu1_0)))


    P_y = get_probability(cost, mu0_0, mu1_0, sigma0_0, tau_prime, pdf0, pdf1)


    decision.set_xdata([tau_prime]*2)


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

tau_slider_ax    = fig.add_axes([0.25, 0.10, 0.45, 0.03], facecolor=axis_color)
tau_slider       = Slider(tau_slider_ax, r'$\tau^,$', -1.0, 1.0, valinit=tau_prime)





def sliders_on_changed(val):
    global mu0_0;     mu0_0      = mu0_slider.val
    global mu1_0;     mu1_0      = mu1_slider.val
    global sigma0_0;  sigma0_0   = sigma0_slider.val
    global sigma1_0;  sigma1_0   = sigma1_slider.val
    global tau_prime; tau_prime  = tau_slider.val

    draw_plots()
mu0_slider.on_changed(sliders_on_changed)
mu1_slider.on_changed(sliders_on_changed)
sigma0_slider.on_changed(sliders_on_changed)
sigma1_slider.on_changed(sliders_on_changed)
tau_slider.on_changed(sliders_on_changed)
sliders_on_changed(None)




reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    sigma0_slider.reset()
    mu0_slider.reset()
    mu1_slider.reset()
    tau_slider.reset()
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
    draw_plots()
def pdf1_radios_on_clicked(label):
    global pdf1; 
    if label == 'normal':
        pdf1 = lambda y_: normal(mu1_0, sigma1_0, y_)
    elif label == 'laplacian':
        pdf1 = lambda y_: laplacian(mu1_0, sigma1_0, y_)
    elif label == 'uniform':
        pdf1 = lambda y_: uniform(mu1_0, sigma1_0, y_)
    draw_plots()
pdf0_radios.on_clicked(pdf0_radios_on_clicked)
pdf1_radios.on_clicked(pdf1_radios_on_clicked)

def txtBox_on_changed(expression, i, j):
    cost[i][j] = float(expression)
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