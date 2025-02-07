from detection_theory import *




## initialize parameters
mu0_0   = -0.25
mu1_0   =  0.25
sigma0_0 =  0.1
sigma1_0 =  0.1
pi_0       = 0.5
alpha_0    = 0.5
cost       = [[0,1],[1,0]]
tau = 0
tau_prime  = 0
t = np.arange(-1.0, 1.0, 0.001)


## define pdf functions
pdf0 = lambda y_: normal(mu0_0, sigma0_0, y_)
pdf1 = lambda y_: normal(mu1_0, sigma1_0, y_)


## set up figure
axis_color = 'lightgoldenrodyellow'
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.40, right=0.75)
xlim_min = -0.1
xlim_max =  1.1
ylim_min = -0.1
ylim_max =  1.1

P_F, P_D = get_Receiver_Operating_Characteristic(cost, mu0_0, mu1_0, sigma0_0, pdf0, pdf1, alpha_0)
[pdf0_line]     = ax.plot(P_F, P_D, linewidth=2, color='red', alpha=0.5)

ax.set_xlim([xlim_min, xlim_max])
ax.set_xlabel(r'$P_F$')
ax.set_ylabel(r'$P_D$')
ax.set_ylim([ylim_min, ylim_max])

def draw_plots():   
    P_F, P_D = get_Receiver_Operating_Characteristic(cost, mu0_0, mu1_0, sigma0_0, pdf0, pdf1, alpha_0)
    pdf0_line.set_xdata(P_F)
    pdf0_line.set_ydata(P_D)

    fig.canvas.draw_idle()




# Sliders
d_slider_ax    = fig.add_axes([0.25, 0.20, 0.45, 0.03], facecolor=axis_color)
d_slider       = Slider(d_slider_ax, r'$d$', -2.0, 2.0, valinit=mu0_0)




def sliders_on_changed(val):
    global mu0_0;   mu0_0     = -d_slider.val
    global mu1_0;   mu1_0     = 0
    global sigma0_0; sigma0_0 = 1
    global sigma1_0; sigma1_0 = sigma0_0

    draw_plots()
d_slider.on_changed(sliders_on_changed)
sliders_on_changed(None)




reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    d_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)



plt.show()