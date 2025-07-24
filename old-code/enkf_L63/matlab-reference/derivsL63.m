function derivs = derivsL63(t,x,par)

%
% Originator: G.J. Hakim, University of Washington
%             hakim@atmos.washington.edu
%
% released under GNU General Public License version 3. http://www.gnu.org/licenses/gpl.html
%
% version control:
% $Date: 2009-02-25 16:04:40 -0800 (Wed, 25 Feb 2009) $
% $Revision: 44 $
% $Author: hakim $
% $Id: derivsL63.m 44 2009-02-26 00:04:40Z hakim $

x_dot  = -par(1)*x(1) + par(1)*x(2);
y_dot  = -x(1)*x(3) + par(2)*x(1) - x(2);
z_dot  =  x(1)*x(2) - par(3)*x(3);
derivs = [x_dot y_dot z_dot]';

