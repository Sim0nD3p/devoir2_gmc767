% Animation of temperature field over time

% Simulation parameters
Lx = 1; Ly = 1;
Nx = 80; Ny = 80;
tf = 1.0;       % total simulation time
dt = 0.0045;    % time-step

dx = Lx / (Nx - 1);
dy = Ly / (Ny - 1);
Nit = floor(tf / dt) + 1;

t = linspace(0, tf, Nit);
x = linspace(0, Lx, Nx);
y = linspace(0, Ly, Ny);
[XX, YY] = meshgrid(x, y);  % MATLAB meshgrid uses 'xy' indexing by default
XX = XX';  % transpose to match Python's indexing='ij'
YY = YY';

% Precompute temperature field (Nx x Ny x Nit)
period = tf / 4;
omega = 2 * pi / period;
temperature = zeros(Nx, Ny, Nit);
for k = 1:Nit
    temperature(:,:,k) = cos(XX) .* cos(YY) .* cos(omega * t(k));
end

% Plotting setup
fig = figure('Position', [100 100 800 800]);
ax = axes;
hold on;

X1 = XX;
Y1 = YY;
contour_values = linspace(-1, 1, 21);
cmap = 'jet';  % MATLAB doesn't have 'bwr' built-in; use a custom one below

% Create a blue-white-red colormap (equivalent to Python's 'bwr')
n = 256;
bwr = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1);
       ones(n/2,1),        linspace(1,0,n/2)', linspace(1,0,n/2)'];
colormap(ax, bwr);

T_init = temperature(:,:,1);
[~, hcf] = contourf(ax, X1, Y1, T_init, contour_values);
[~, hc]  = contour(ax, X1, Y1, T_init, contour_values, 'LineColor', 'k');
clim([-1 1]);
cb = colorbar;
cb.Ticks = linspace(-1, 1, 11);
cb.Label.String = '\phi';

xlim([0 1]);
ylim([0 1]);
xlabel('x');
ylabel('y');

time_text = text(0.02, 0.95, '', 'Units', 'normalized', ...
    'FontSize', 14, 'VerticalAlignment', 'top', ...
    'BackgroundColor', [0.96 0.87 0.70], 'EdgeColor', 'k', ...
    'Margin', 4);

% Animation loop — save to GIF
frames = Nit;
video_time = 5.0; % seconds
fps = round(frames / video_time);
delay_time = 1 / fps;
filename = 'Temperature_evolution_example.gif';

fprintf('fps = %d\n', fps);

for i = 1:frames
    % Clear previous contours
    cla(ax);

    Z = temperature(:,:,i);
    [~, hcf] = contourf(ax, X1, Y1, Z, contour_values);
    hold on;
    [~, hc]  = contour(ax, X1, Y1, Z, contour_values, 'LineColor', 'k');
    clim([-1 1]);
    colormap(ax, bwr);

    time_text = text(0.02, 0.95, sprintf('Time: %.4f s', t(i)), ...
        'Units', 'normalized', 'FontSize', 14, ...
        'VerticalAlignment', 'top', ...
        'BackgroundColor', [0.96 0.87 0.70], 'EdgeColor', 'k', ...
        'Margin', 4);

    xlim([0 1]);
    ylim([0 1]);
    xlabel('x');
    ylabel('y');

    drawnow;

    % Capture frame and write to GIF
    frame = getframe(fig);
    im = frame2im(frame);
    [A, map] = rgb2ind(im, 256);

    if i == 1
        imwrite(A, map, filename, 'gif', 'LoopCount', 0, 'DelayTime', delay_time);
    else
        imwrite(A, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', delay_time);
    end
end

fprintf('GIF saved to %s\n', filename);